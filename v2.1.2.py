import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
# ==========================================
# 1. GLOBAL SETTINGS & HYPERPARAMETERS
# ==========================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

EMBED_DIM = 768       
NUM_HEADS = 12        # 768 / 12 = 64 (head_dim is 64)
NUM_LAYERS = 6        
BLOCK_SIZE = 256      
BATCH_SIZE = 16       
BATCHES_PER_STEP = 4
WEIGHT_DECAY = 0.005
TOTAL_STEPS = 2000    
EVAL_INTERVAL = 200  

# ==========================================
# 2. UNIVERSAL HARDWARE ENGINE SELECTOR
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")          
elif torch.backends.mps.is_available():
    device = torch.device("mps")           
else:
    device = torch.device("cpu")           

print(f"🚀 INITIALIZING ENGINE ON: {device}")

# ==========================================
# 3. TEXT LOADING & TOKENIZER MANAGEMENT
# ==========================================
file_path = "tinystories.txt"
TOKENIZER_DIR = "tokenizer_model"

with open(file_path, "r", encoding="utf-8") as f:
    corpus_text = f.read()

if os.path.exists(TOKENIZER_DIR):
    print("💾 Loading existing persistent tokenizer from disk...")
    tokenizer = ByteLevelBPETokenizer.from_file(
        vocab_filename=f"{TOKENIZER_DIR}/vocab.json", 
        merges_filename=f"{TOKENIZER_DIR}/merges.txt"
    )
else:
    print("⏳ No tokenizer found. Training a fresh pipeline...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    with open("temp_corpus.txt", "w", encoding="utf-8") as t:
        t.write(corpus_text)
        
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(["temp_corpus.txt"], vocab_size=8192, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
    tokenizer.save_model(TOKENIZER_DIR)
    os.remove("temp_corpus.txt")
    print(f"✅ Tokenizer saved permanently to '{TOKENIZER_DIR}/'")

vocab = tokenizer.get_vocab()
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_IDX, EOS_IDX = vocab["<pad>"], vocab["</s>"]

# ==========================================
# 4. HIGH-PERFORMANCE PINNED MEMORY DATA PIPELINE
# ==========================================
print("📦 Preparing memory-mapped disk storage chunks...")
corpus_tokens = tokenizer.encode(corpus_text).ids
split_idx = int(0.9 * len(corpus_tokens))

train_filename = "train.bin"
val_filename = "val.bin"

if not os.path.exists(train_filename) or not os.path.exists(val_filename):
    train_ids = np.array(corpus_tokens[:split_idx], dtype=np.uint16)
    val_ids = np.array(corpus_tokens[split_idx:], dtype=np.uint16)
    train_ids.tofile(train_filename)
    val_ids.tofile(val_filename)
    del train_ids, val_ids

del corpus_text, corpus_tokens  # Clear large variables completely out of RAM

train_data = np.memmap(train_filename, dtype=np.uint16, mode='r')
val_data = np.memmap(val_filename, dtype=np.uint16, mode='r')

def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    
    # Extract slices directly from disk using CPU
    x_list = [torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix]
    y_list = [torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix]
    
    x = torch.stack(x_list).to(device, non_blocking=True)
    y = torch.stack(y_list).to(device, non_blocking=True)
    return x, y

# ==========================================
# 5. CORE LLAMACLONE ARCHITECTURE (FROM SCRATCH)
# ==========================================
class RMSNorm(nn.Module):
    """Root Mean Square Normalization for computationally lightweight scaling"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) precomputed rotation cache"""
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq) 
        emb = torch.cat((freqs, freqs), dim=-1) 
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        return (
            self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1),
            self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1),
        )

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class FlashMultiHeadAttention(nn.Module):
    """Kernel-fused Scaled Dot-Product Attention utilizing RoPE"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(Q, T)
        cos, sin = cos.to(x.device), sin.to(x.device)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # Enforces highly optimized low-level fused FlashAttention layout
        context_mapped = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        return self.W_out(context_mapped.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit for rapid learning convergence"""
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.W_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_down = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_up = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W_up(F.silu(self.W_gate(x)) * self.W_down(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = FlashMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        
        # Calculate standard 8/3 hidden dimension layout 
        hidden_dim = int(2 * (embed_dim * 4) / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  
        self.ffn = SwiGLU(embed_dim, hidden_dim, dropout)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class PureScratchTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm_final = RMSNorm(embed_dim)
        self.W_output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.W_output.weight = self.word_embeddings.weight 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.dropout(self.word_embeddings(input_ids))
        for block in self.blocks: 
            x = block(x)
        return self.W_output(self.norm_final(x))


model = PureScratchTransformer(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=0.1
).to(device)

# ==========================================
# 6. PIPELINED TRAINING ENGINE
# ==========================================
def get_lr_multiplier(step, warmup_steps=200, total_steps=2000):
    """Calculates a learning rate multiplier for linear warmup."""
    if step < warmup_steps:
        # Linear warmup phase (0.0 to 1.0)
        return float(step) / float(max(1, warmup_steps))
    return 1.0

def train_scratch_model():
    weight_file = "scratch_weights.pt"
    if os.path.exists(weight_file):
        print("🔄 Loading existing stable weights...")
        model.load_state_dict(torch.load(weight_file, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=WEIGHT_DECAY, foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',       # We want to minimize loss
        factor=0.5,       # Multiply LR by 0.5 when loss plateaus (cut in half)
        patience=2,       # Wait for 2 eval intervals with no improvement before dropping LR
    )
    best_val_loss = float('inf')

    print(f"📊 Starting execution stream for {TOTAL_STEPS} steps...")
    step_start = time.time()
    total_train_loss = 0
    progress_bar = tqdm(total=TOTAL_STEPS, desc="🧠 Training Progress", unit="step")

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        batch_x, batch_y = get_batch("train")
        
        # 1. Reset gradients only at the beginning of a step
        if (step - 1) % BATCHES_PER_STEP == 0:
            optimizer.zero_grad(set_to_none=True)
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu' 
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
            logits = model(batch_x)
            # 2. Divide loss down by the accumulation factor
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_y.view(-1))
            loss = loss / BATCHES_PER_STEP

        loss.backward()
        
        # 3. Only step weights forward when a full macro-batch sequence ends
        if step % BATCHES_PER_STEP == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Only manually override the learning rate during the active warmup phase
            if step < 200:
                lr_scale = get_lr_multiplier(step, warmup_steps=200, total_steps=TOTAL_STEPS)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0006 * lr_scale
                    
            optimizer.step()
        
        # Track logging accurately using the original unscaled metric value
        total_train_loss += (loss.item() * BATCHES_PER_STEP)
        progress_bar.set_postfix({"loss": f"{(loss.item() * BATCHES_PER_STEP):.4f}"})
        progress_bar.update(1)

        if step % EVAL_INTERVAL == 0:
            avg_train_loss = total_train_loss / EVAL_INTERVAL
            progress_bar.write("\n⏳ Running validation check...")
            
            model.eval()
            total_val_loss = 0
            val_iters = 20  
            with torch.inference_mode():
                for _ in range(val_iters):
                    val_x, val_y = get_batch("val")
                    val_logits = model(val_x)
                    total_val_loss += F.cross_entropy(val_logits.view(-1, VOCAB_SIZE), val_y.view(-1)).item()
            
            avg_val_loss = total_val_loss / val_iters
            scheduler.step(avg_val_loss)
            progress_bar.write(
                f"► Step {step:04d}/{TOTAL_STEPS} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Interval Time: {time.time() - step_start:.2f}s"
            )
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), weight_file)
                progress_bar.write("   [SAVED BEST MODEL CONFIGURATION]")
                
            total_train_loss = 0
            step_start = time.time()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            progress_bar.write("=" * 50 + "\n")

    progress_bar.close()

# ==========================================
# 7. SAMPLING & GENERATION ENGINE
# ==========================================
def top_p_filtering(logits, top_p=0.85, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def run_scratch_generation(prompt, max_new_tokens=50, temperature=0.8, top_p=0.85, repetition_penalty=1.05):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    generated_tokens, start_time, printed_len = [], time.time(), len(prompt)
    
    print(f"\n--- PROMPT: {prompt} ---")
    print(prompt, end="", flush=True)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            next_token_logits = model(input_ids[:, -BLOCK_SIZE:])[0, -1, :].clone().float()

            for token_id in set(generated_tokens[-15:]):
                if next_token_logits[token_id] > 0: next_token_logits[token_id] /= repetition_penalty
                else: next_token_logits[token_id] *= repetition_penalty

            next_token_id = torch.multinomial(F.softmax(top_p_filtering(next_token_logits / temperature, top_p=top_p), dim=-1), num_samples=1)
            if next_token_id.item() == EOS_IDX: break

            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            full_decoded = tokenizer.decode(input_ids[0].tolist())
            print(full_decoded[printed_len:], end="", flush=True)
            printed_len = len(full_decoded)

    print(f"\n[Perf Stat: {len(generated_tokens) / (time.time() - start_time):.2f} tokens/sec]")

# ==========================================
# 8. EXECUTION GATEWAY
# ==========================================
if __name__ == "__main__":
    import gc
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()
    gc.collect()
    
    train_scratch_model()
    
    for p in ["The cat and the dog are", "I love fruits and"]:
        run_scratch_generation(p)
        print("\n" + "="*50)