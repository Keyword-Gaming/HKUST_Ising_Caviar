import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

# ==========================================
# 1. GLOBAL SETTINGS & HYPERPARAMETERS
# ==========================================
# Suppress Hugging Face download/symlink environment noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Network Architecture Specs
EMBED_DIM = 768       # Standard "Base" size (or 512 if you want to play it safe)
NUM_HEADS = 12        # EMBED_DIM must be perfectly divisible by NUM_HEADS (768 / 12 = 64)
NUM_LAYERS = 6        # Deeper network allows for more complex grammar rules
BLOCK_SIZE = 256      # Crucial! Gives the model a 4x longer memory window
BATCH_SIZE = 64       # Dropped from 128 to offset the memory increase from BLOCK_SIZE
# Optimization Specs
WEIGHT_DECAY = 0.005
TOTAL_STEPS = 2000    # Total training steps (replaces slow epoch-based loops)
EVAL_INTERVAL = 200  # How often to check validation loss and save weights

# ==========================================
# 2. UNIVERSAL HARDWARE ENGINE SELECTOR
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")          # Standard Windows/Linux NVIDIA GPUs
elif torch.backends.mps.is_available():
    device = torch.device("mps")           # Apple Silicon M-Series GPUs (Mac)
else:
    device = torch.device("cpu")           # Fallback safety safety mechanism

print(f"🚀 INITIALIZING ENGINE ON: {device}")

# ==========================================
# 3. TEXT LOADING & TOKENIZER MANAGEMENT
# ==========================================
file_path = "tinystories.txt"
TOKENIZER_DIR = "tokenizer_model"

with open(file_path, "r", encoding="utf-8") as f:
    corpus_text = f.read()

# Safe Save/Load Tokenizer Pipeline
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
# 4. HIGH-PERFORMANCE NATIVE GPU PIPELINE
# ==========================================
# Tokenize and push entire corpus to GPU immediately to eliminate CPU-to-GPU data lag
corpus_tokens = tokenizer.encode(corpus_text).ids
split_idx = int(0.9 * len(corpus_tokens))

train_data = torch.tensor(corpus_tokens[:split_idx], dtype=torch.long, device=device)
val_data = torch.tensor(corpus_tokens[split_idx:], dtype=torch.long, device=device)

def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,), device=device) # Explicitly match device here too just in case
    
    # Create a 2D grid of indices natively on the GPU
    start_positions = ix.unsqueeze(1) # Shape: (BATCH_SIZE, 1)
    offsets = torch.arange(BLOCK_SIZE, device=device) # Shape: (BLOCK_SIZE,)
    
    grid_indices = start_positions + offsets # Shape: (BATCH_SIZE, BLOCK_SIZE)
    
    x = data[grid_indices]
    y = data[grid_indices + 1] 
    return x, y

# ==========================================
# 5. CORE NEURAL NETWORK ARCHITECTURE
# ==========================================
class PositionalEncodingFromScratch(nn.Module):
    def __init__(self, embed_dim, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x): 
        return x + self.pe[:, :x.size(1)]

class PureMultiHeadAttentionFromScratch(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights = None

    def forward(self, x, retain_attention=False):
        B, T, C = x.shape
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if retain_attention:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            attn_weights = F.softmax(scores + mask, dim=-1)
            self.last_attention_weights = attn_weights[0].mean(dim=0).detach().cpu()
            context_mapped = torch.matmul(self.dropout(attn_weights), V)
        else:
            context_mapped = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
            )

        return self.W_out(context_mapped.transpose(1, 2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = PureMultiHeadAttentionFromScratch(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(approximate='tanh'), 
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x, retain_attention=False):
        x = x + self.attention(self.norm1(x), retain_attention=retain_attention)
        x = x + self.ffn(self.norm2(x))
        return x

class PureScratchTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncodingFromScratch(embed_dim, max_len=max(2048, BLOCK_SIZE))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm_final = nn.LayerNorm(embed_dim)
        self.W_output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.W_output.weight = self.word_embeddings.weight 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, retain_attention=False):
        x = self.dropout(self.pos_encoder(self.word_embeddings(input_ids)))
        for block in self.blocks: x = block(x, retain_attention=retain_attention)
        return self.W_output(self.norm_final(x))

# Instantiate and bind the model to the active computing engine
model = PureScratchTransformer(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=0.1
).to(device)

# ==========================================
# 6. PIPELINED TRAINING ENGINE
# ==========================================
def train_scratch_model():
    weight_file = "scratch_weights.pt"
    if os.path.exists(weight_file):
        print("🔄 Loading existing stable weights (cross-device mapped)...")
        model.load_state_dict(torch.load(weight_file, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=WEIGHT_DECAY, foreach=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
    best_val_loss = float('inf')

    print(f"📊 Starting unified execution stream for {TOTAL_STEPS} steps...")
    step_start = time.time()
    total_train_loss = 0
    # Initialize a master progress bar for the entire training run
    progress_bar = tqdm(total=TOTAL_STEPS, desc="🧠 Training Progress", unit="step")

    # 1. Determine the optimal data type for your hardware
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        batch_x, batch_y = get_batch("train")
        
        optimizer.zero_grad(set_to_none=True)
        
        # 2. Wrap your forward pass in PyTorch's Autocast context
        # Note: Use device_type="cuda" or device_type="cpu" (MPS uses 'cpu' autocast or native support depending on version, 
        # but for universal compatibility, you can check device.type)
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu' 
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_train_loss += loss.item()
        
        # Update the progress bar by 1 step, displaying the live loss value
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        progress_bar.update(1)

        # Interval Validation Evaluation Block
        if step % EVAL_INTERVAL == 0:
            avg_train_loss = total_train_loss / EVAL_INTERVAL
            
            # Pause progress bar printing layout momentarily to log evaluation stats cleanly
            progress_bar.write("\n⏳ Running mid-training validation check...")
            
            # Validation Step
            model.eval()
            total_val_loss = 0
            val_iters = 20  
            with torch.inference_mode():
                for _ in range(val_iters):
                    val_x, val_y = get_batch("val")
                    val_logits = model(val_x)
                    total_val_loss += F.cross_entropy(val_logits.view(-1, VOCAB_SIZE), val_y.view(-1)).item()
            
            avg_val_loss = total_val_loss / val_iters
            
            # Write out metrics using progress_bar.write to prevent breaking the slider bar layout
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
            progress_bar.write("=" * 50 + "\n")

    # Cleanly close the visual display layout when complete
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
            next_token_logits = model(input_ids[:, -BLOCK_SIZE:], retain_attention=False)[0, -1, :].clone().float()

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    train_scratch_model()
    for p in ["the sky is made of", "during the twentieth century, the global economy", "there are six legs on a", "four times six is larger than", "Steve Harvey performs on the famous game show", "The cat is", "The dog is", "The cat and the dog are","He eats the","Notably,","Chinese Businessman Jack Ma ate"," ","I love fruits and"]:
        run_scratch_generation(p)
        print("\n" + "="*50)