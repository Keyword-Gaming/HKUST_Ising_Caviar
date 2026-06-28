import os
import gc
import time
import math
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

VOCAB_SIZE = 8192
EMBED_DIM = 768       
NUM_HEADS = 12        # 768 / 12 = 64 (head_dim is 64)
NUM_LAYERS = 6        
BLOCK_SIZE = 128      
BATCH_SIZE = 16       
BATCHES_PER_STEP = 8
WEIGHT_DECAY = 0.005
TOTAL_STEPS = 2000    
EVAL_INTERVAL = 100  
LEARNING_RATE = 0.00001
WARMUP_TIME = EVAL_INTERVAL * BATCHES_PER_STEP #steps

# ==========================================
# 2. UNIVERSAL HARDWARE ENGINE SELECTOR
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")          
elif torch.backends.mps.is_available():
    device = torch.device("mps")           
else:
    device = torch.device("cpu")           

print(f"🚀 INITIALIZING v2.1.4 ENGINE ON: {device}")

# ==========================================
# 3. TEXT LOADING & TOKENIZER MANAGEMENT
# ==========================================
import pandas as pd

train_csv_path = "train.csv"
val_csv_path = "validation.csv"
TOKENIZER_DIR = "tokenizer_model"

# Let's read just the training CSV first to build the tokenizer if it doesn't exist
print("📖 Loading Kaggle data files...")
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Kaggle TinyStories usually names the text column 'text' or 'story'
# Double-check your CSV columns and update this string if it's named differently!
text_column = "text" 

if os.path.exists(TOKENIZER_DIR):
    print("💾 Loading existing persistent tokenizer from disk...")
    tokenizer = ByteLevelBPETokenizer.from_file(
        vocab_filename=f"{TOKENIZER_DIR}/vocab.json", 
        merges_filename=f"{TOKENIZER_DIR}/merges.txt"
    )
else:
    print("⏳ No tokenizer found. Training a fresh pipeline on training data...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    
    # Dump raw text to a temp file for tokenizers library to ingest
    with open("temp_corpus.txt", "w", encoding="utf-8") as t:
        for text in train_df[text_column].dropna():
            t.write(str(text) + "\n")
        
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(["temp_corpus.txt"], vocab_size=VOCAB_SIZE, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
    tokenizer.save_model(TOKENIZER_DIR)
    os.remove("temp_corpus.txt")
    print(f"✅ Tokenizer saved permanently to '{TOKENIZER_DIR}/'")

vocab = tokenizer.get_vocab()
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_IDX, EOS_IDX = vocab["<pad>"], vocab["</s>"]

# ==========================================
# 4. HIGH-PERFORMANCE PINNED MEMORY DATA PIPELINE (KAGGLE CSV EDITION)
# ==========================================
print("📦 Compiling Kaggle CSV records into binary memory maps...")

train_filename = "train.bin"
val_filename = "val.bin"

# Helper function to convert dataframe column into a massive continuous token string
def process_dataframe_to_bin(df, output_filename):
    all_token_ids = []
    # Drop rows that don't contain a string to keep formatting safe
    for story in tqdm(df[text_column].dropna(), desc=f"Encoding {output_filename}"):
        story_clean = str(story).strip()
        if story_clean:
            # Wrap each Kaggle row natively in start/stop tokens
            tokens = tokenizer.encode(f"<s> {story_clean} </s>").ids
            all_token_ids.extend(tokens)
            
    # Write directly to high-performance disk binary
    np_ids = np.array(all_token_ids, dtype=np.uint16)
    np_ids.tofile(output_filename)
    del np_ids, all_token_ids

# Process both datasets completely separately to prevent data leakage
if os.path.exists(train_filename):
    print("📁 Training file loaded!")
else:
    print("❌ Training data not found.")
    process_dataframe_to_bin(train_df, train_filename)

if os.path.exists(val_filename):
    print("📁 Validation file loaded!")
else:
    print("❌ Validation data not found.")
    process_dataframe_to_bin(val_df, val_filename)

# Aggressive cleanup of dataframes from system RAM before starting PyTorch
del train_df, val_df 
gc.collect()

# Remap clean bin streams
train_data = np.memmap(train_filename, dtype=np.uint16, mode='r')
val_data = np.memmap(val_filename, dtype=np.uint16, mode='r')

def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    # Generate random start positions as a NumPy array
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)).numpy()
    
    # Broadcast into a 2D grid of indices natively in NumPy
    start_positions = ix[:, None]                  # Shape: (BATCH_SIZE, 1)
    offsets = np.arange(BLOCK_SIZE)                # Shape: (BLOCK_SIZE,)
    grid_indices = start_positions + offsets       # Shape: (BATCH_SIZE, BLOCK_SIZE)
    
    # Slice the entire batch out of the memmap at once
    x = torch.from_numpy(data[grid_indices].astype(np.int64)).to(device, non_blocking=True)
    y = torch.from_numpy(data[grid_indices + 1].astype(np.int64)).to(device, non_blocking=True)
    return x, y

# ==========================================
# 5. CORE LLAMACLONE ARCHITECTURE (FROM SCRATCH)
# ==========================================
class RMSNorm(nn.Module):
    """Root Mean Square Normalization for computationally lightweight scaling"""
    def __init__(self, dim, eps=1e-5):
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
    def __init__(self, embed_dim, num_heads, dropout=0.35):
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
    def __init__(self, embed_dim, hidden_dim, dropout=0.35):
        super().__init__()
        self.W_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_down = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.W_up = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W_up(F.silu(self.W_gate(x)) * self.W_down(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.35):
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
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout=0.35):
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
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=0.35
).to(device)

# ==========================================
# 6. PIPELINED TRAINING ENGINE
# ==========================================
def get_lr_multiplier(step, warmup_steps):
    """Calculates a learning rate multiplier for linear warmup."""
    if step < warmup_steps:
        # Linear warmup phase (0.0 to 1.0)
        return float(step) / float(max(1, warmup_steps))
    return 1.0

def calculate_accuracy(logits, targets, top_k=1):
    """Calculates the percentage of tokens where the target fell within the top k predictions."""
    # Reshape to (Total Tokens, Vocab Size)
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # Get the indices of the highest k values
    _, top_indices = logits.topk(top_k, dim=-1)
    
    # Check if target is in top k choices along the last dimension
    correct = top_indices.eq(targets.unsqueeze(-1).expand_as(top_indices))
    return correct.any(dim=-1).float().mean().item()

def train_scratch_model():
    global LEARNING_RATE, WARMUP_TIME
    weight_file = "scratch_weights.pt"
    if os.path.exists(weight_file):
        print("🔄 Loading existing stable weights...")
        model.load_state_dict(torch.load(weight_file, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, foreach=True if torch.cuda.is_available() else False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',       
        factor=0.5,       
        patience=2,       
    )
    best_val_loss = float('inf')

    print(f"📊 Starting execution stream for {TOTAL_STEPS} steps...")
    total_train_loss = 0
    progress_bar = tqdm(total=TOTAL_STEPS, desc="🧠 Training Progress", unit="step")

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        batch_x, batch_y = get_batch("train")
        
        if (step - 1) % BATCHES_PER_STEP == 0:
            optimizer.zero_grad(set_to_none=True)
        
        if torch.cuda.is_available():
            device_type = 'cuda'
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            use_autocast = True
        elif torch.backends.mps.is_available():
            device_type = 'mps'
            use_autocast = False
        else:
            device_type = 'cpu'
            use_autocast = False
        
        if use_autocast:
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                logits = model(batch_x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_y.view(-1)) / BATCHES_PER_STEP
        else:
            # Safe, uncorrupted full-precision path for Mac GPU
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_y.view(-1)) / BATCHES_PER_STEP

        loss.backward()
        
        if step % BATCHES_PER_STEP == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Warmup Phase (Steps 1 to 200)
            if step <= WARMUP_TIME:
                lr_scale = get_lr_multiplier(step, warmup_steps=WARMUP_TIME)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2 * LEARNING_RATE * lr_scale  
            # Post-Warmup Stable Phase (Step 201+)
            else:
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > LEARNING_RATE: 
                        param_group['lr'] = LEARNING_RATE  
                    
            optimizer.step()
        
        total_train_loss += (loss.item() * BATCHES_PER_STEP)
        progress_bar.set_postfix({"loss": f"{(loss.item() * BATCHES_PER_STEP):.4f}"})
        progress_bar.update(1)

        if step % EVAL_INTERVAL == 0:
            avg_train_loss = total_train_loss / EVAL_INTERVAL
            progress_bar.write("⏳ Running comprehensive validation check...")
            
            model.eval()
            total_val_loss = 0
            total_top1_acc = 0
            total_top5_acc = 0
            val_iters = 100  
            
            # Separate progress bar to track the evaluation stage cleanly
            val_bar = tqdm(total=val_iters, desc="🔍 Validating", unit="batch", leave=False)
            
            with torch.inference_mode():
                for _ in range(val_iters):
                    val_x, val_y = get_batch("val")
                    val_logits = model(val_x)
                    
                    loss_val = F.cross_entropy(val_logits.view(-1, VOCAB_SIZE), val_y.view(-1))
                    total_val_loss += loss_val.item()
                    
                    total_top1_acc += calculate_accuracy(val_logits, val_y, top_k=1)
                    total_top5_acc += calculate_accuracy(val_logits, val_y, top_k=5)
                    val_bar.update(1)
            
            val_bar.close() # Close evaluation bar smoothly out of terminal lines
            
            avg_val_loss = total_val_loss / val_iters
            avg_top1 = (total_top1_acc / val_iters) * 100   
            avg_top5 = (total_top5_acc / val_iters) * 100   
            val_perplexity = math.exp(avg_val_loss)          
            
            scheduler.step(avg_val_loss)
            
            #Extract active learning rate directly from the parameter group
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print a detailed multi-line snapshot of model health
            progress_bar.write("-" * 50)
            progress_bar.write(f"📊 EVALUATION CHECKPOINT | Step {step:04d}/{TOTAL_STEPS}")
            progress_bar.write(f"• Learning Rate: {current_lr:.8f}")  # Prints full precision profile
            progress_bar.write(f"• Training Loss:    {avg_train_loss:.4f}")
            progress_bar.write(f"• Validation Loss:      {avg_val_loss:.4f}")
            progress_bar.write(f"• Perplexity:    {val_perplexity:.2f} (lower is better)")
            progress_bar.write(f"• Top-1 Accuracy:     {avg_top1:.2f}% (exact guess match)")
            progress_bar.write(f"• Top-5 Accuracy:     {avg_top5:.2f}% (correct word in top 5 options)")
            print("\n🎬 [Live Progress Sample]")
            # Use a simple, iconic TinyStories style prompt
            run_scratch_generation(
                prompt="Once upon a time, a little girl named Lily found a", 
                max_new_tokens=80, 
                temperature=0.8,
                top_p=0.9
            )
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), weight_file)
                progress_bar.write("   [SAVED BEST MODEL CONFIGURATION]")
            
            print("="*50 + "\n")
            # Switch back to training mode so gradients keep updating
            model.train()

            total_train_loss = 0
            if device_type == "cuda": torch.cuda.empty_cache()
            elif device_type == "mps": torch.mps.empty_cache()

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
def run_scratch_generation(prompt, max_new_tokens=50, temperature=0.85, top_p=0.9, repetition_penalty=1.25):
    model.eval()

    # We append the text, then grab its IDs.
    tokenized_prompt = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokenized_prompt], dtype=torch.long, device=device)
    
    generated_tokens = []
    start_time = time.time()
    
    print(f"\n--- PROMPT: {prompt} ---")
    print(prompt, end="", flush=True)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Crop to the context window safely
            cond_ids = input_ids[:, -BLOCK_SIZE:]
            next_token_logits = model(cond_ids)[0, -1, :].clone().float()

            # Apply repetition penalty dynamically
            for token_id in set(generated_tokens[-15:]):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= repetition_penalty
                else:
                    next_token_logits[token_id] *= repetition_penalty

            # Scale and filter logits
            filtered_logits = top_p_filtering(next_token_logits / max(temperature, 1e-5), top_p=top_p)
            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            
            if next_token_id.item() == EOS_IDX: 
                break

            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Just decode the single new token that was generated!
            # This avoids calculating string indices and guarantees no stuttering.
            single_token_text = tokenizer.decode([next_token_id.item()])
            print(single_token_text, end="", flush=True)
            
    print(f"\n[Perf Stat: {len(generated_tokens) / (time.time() - start_time):.2f} tokens/sec]")
# ==========================================
# 8. EXECUTION GATEWAY
# ==========================================
if __name__ == "__main__":
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()
    gc.collect()
    
    descision = input("Train Model or Print Prompt (T or P)")
    if descision[0].upper() == "T":
        train_scratch_model()
        for p in ["The cat and the dog are", "I love fruits and", "Over by the shadows", "The three largest donkeys", "two plus two equals", "A small woods housed", "Archie is a child and", "For breakfast there will be", "Beekeepers united for the"]:
            run_scratch_generation(p)
            print("\n" + "="*50)
    elif descision[0].upper() == "P":
        prompt = input("What sentence would you like the AI to finish?: ")
        print("\n" + "="*50)
        run_scratch_generation(prompt)
        print("\n" + "="*50)
    input("Press enter to exit program: ")