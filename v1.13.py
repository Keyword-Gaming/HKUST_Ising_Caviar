import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Setup ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 INITIALIZING V14 STABILIZED ENGINE ON: {device}")

# --- Data Loading ---
file_path = "essays.txt"
with open(file_path, "r", encoding="utf-8") as f:
    corpus_text = f.read()

from tokenizers import ByteLevelBPETokenizer
TOKENIZER_DIR = "tokenizer_model"
file_path = "essays.txt"

# 1. Read the raw text corpus
with open(file_path, "r", encoding="utf-8") as f:
    corpus_text = f.read()

# 2. Initialize or load the tokenizer safely
if os.path.exists(TOKENIZER_DIR):
    print("💾 Loading existing tokenizer from disk...")
    tokenizer = ByteLevelBPETokenizer.from_file(
        vocab_filename=f"{TOKENIZER_DIR}/vocab.json", 
        merges_filename=f"{TOKENIZER_DIR}/merges.txt"
    )
else:
    print("⏳ No tokenizer found. Training a new one...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    
    # Write a temporary file for training
    with open("temp_corpus.txt", "w", encoding="utf-8") as t:
        t.write(corpus_text)
        
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        ["temp_corpus.txt"], 
        vocab_size=2048, 
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )
    
    # Save the tokenizer so we never have to retrain on this dataset
    tokenizer.save_model(TOKENIZER_DIR)
    os.remove("temp_corpus.txt")
    print(f"✅ Tokenizer trained and saved to '{TOKENIZER_DIR}/'")

# Setup vocabulary dimensions
vocab = tokenizer.get_vocab()
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_IDX, EOS_IDX = vocab["<pad>"], vocab["</s>"]

corpus_tokens = tokenizer.encode(corpus_text).ids
split_idx = int(0.9 * len(corpus_tokens))
train_tokens, val_tokens = corpus_tokens[:split_idx], corpus_tokens[split_idx:]

class ScratchTokenDataset(Dataset):
    def __init__(self, tokens, block_size, device):
        # Move ALL tokens to the GPU right here, once!
        self.tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        self.block_size = block_size
        
    def __len__(self):
        return max(0, len(self.tokens_tensor) - self.block_size)

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.block_size
        
        # Slicing happens directly inside GPU memory now
        x = self.tokens_tensor[start_idx : end_idx]
        y = self.tokens_tensor[start_idx + 1 : end_idx + 1]
        
        return x, y

# --- Hyperparameters ---
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 4
BLOCK_SIZE = 64
BATCH_SIZE = 128
DROPOUT = 0.1
WEIGHT_DECAY = 0.01

# --- Core Architecture ---
class PositionalEncodingFromScratch(nn.Module):
    def __init__(self, embed_dim, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

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
        self.pos_encoder = PositionalEncodingFromScratch(embed_dim, max_len=256)
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

model = PureScratchTransformer(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT
).to(device)

# --- Training Loop ---
def train_scratch_model(epochs=100):
    weight_file = "scratch_weights.pt"
    if os.path.exists(weight_file):
        print("🔄 Loading existing stable weights...")
        model.load_state_dict(torch.load(weight_file, map_location=device))

    train_loader = DataLoader(
        ScratchTokenDataset(train_tokens, block_size=BLOCK_SIZE, device=device), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        ScratchTokenDataset(val_tokens, block_size=BLOCK_SIZE, device=device), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        drop_last=True
    )
    # FIX: Reverted to ultra-stable foreach execution to prevent Apple Silicon driver crashes
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=WEIGHT_DECAY, foreach=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float('inf')

    print(f"📊 Total train batches per epoch: {len(train_loader)}")

    for epoch in range(epochs):
        model.train()
        total_train_loss, step_start = 0, time.time()
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            
            optimizer.zero_grad(set_to_none=True)
            
            # FIX: Dropped buggy autocast block. Running pure FP32 for maximum numerical stability.
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_y.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"   [Epoch {epoch+1}] Batch {batch_idx:03d}/{len(train_loader)} | Loss: {loss.item():.4f} | Time/200: {time.time() - step_start:.2f}s")
                step_start = time.time()

        scheduler.step()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.inference_mode():
            for val_x, val_y in val_loader:
                total_val_loss += F.cross_entropy(model(val_x.to(device)).view(-1, VOCAB_SIZE), val_y.to(device).view(-1)).item()
        
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"► Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss and avg_val > 0:
            best_val_loss = avg_val
            torch.save(model.state_dict(), weight_file)
            print("   [SAVED BEST MODEL]")
        if batch_idx >= 1000: 
            print("Halting epoch early to test generation...")
            break

# --- Sampling & Exec ---
def top_p_filtering(logits, top_p=0.85, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def run_scratch_generation(prompt, max_new_tokens=50, temperature=0.8, top_p=0.85, repetition_penalty=1.3):
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

if __name__ == "__main__":
    train_scratch_model(epochs=10)
    for p in ["the sky is made of", "during the twentieth century, the global economy", "there are six legs on a", "four times six is larger than", "Steve Harvey performs on the famous game show", "The cat is", "The dog is", "The cat and the dog are","He eats the","Notably,","Chinese Businessman Jack Ma ate"," ","I love fruits and"]:
        run_scratch_generation(p)
        print("\n" + "="*50)