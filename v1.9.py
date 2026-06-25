import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Environment & Hardware Setup ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Explicitly default to Apple Silicon acceleration (MPS) when available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using deep multi-head transformer engine on hardware: {device}")

# --- Data Preparation ---
file_path = "essays.txt"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    print(f"Loaded {len(corpus_text)} characters from '{file_path}'!")
else:
    raise FileNotFoundError(f"Missing '{file_path}'. Please verify your data generator script setup.")

# --- Fluency Fix: Byte-Level BPE Tokenizer Setup ---
from tokenizers import ByteLevelBPETokenizer

# Initialize a professional byte-level BPE tokenizer (handles spaces natively)
tokenizer = ByteLevelBPETokenizer()

with open("temp_corpus.txt", "w", encoding="utf-8") as t:
    t.write(corpus_text)

# Train the tokenizer on your dataset
tokenizer.train(["temp_corpus.txt"], vocab_size=2000, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>"
])
os.remove("temp_corpus.txt")

vocab = tokenizer.get_vocab()
VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_IDX = vocab["<pad>"]
EOS_IDX = vocab["</s>"]
print(f"Byte-Level Vocabulary size: {VOCAB_SIZE} unique pieces.")

corpus_tokens = tokenizer.encode(corpus_text).ids
print(f"Corpus subword token count: {len(corpus_tokens)}")

# --- Data Splitting (Train vs Validation Track) ---
split_idx = int(0.9 * len(corpus_tokens))
train_tokens = corpus_tokens[:split_idx]
val_tokens = corpus_tokens[split_idx:]

# --- PyTorch Dataset Definition ---
class ScratchTokenDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size]
        target = self.tokens[idx + 1 : idx + self.block_size + 1]
        return torch.tensor(chunk, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# --- Hyperparameters ---
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
BLOCK_SIZE = 64
BATCH_SIZE = 16
DROPOUT = 0.1

# --- Transformer Core Architecture From Scratch ---
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
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

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

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask

        attention_weights = F.softmax(scores, dim=-1)
        if retain_attention:
            self.last_attention_weights = attention_weights[0].mean(dim=0).detach().cpu()

        attention_weights = self.dropout(attention_weights)
        context_mapped = torch.matmul(attention_weights, V)
        context_unified = context_mapped.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_out(context_unified)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = PureMultiHeadAttentionFromScratch(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
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
        self.W_output = nn.Linear(embed_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, retain_attention=False):
        x = self.word_embeddings(input_ids)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, retain_attention=retain_attention)
        x = self.norm_final(x)
        return self.W_output(x)

# Initialize Model Engine
model = PureScratchTransformer(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT
).to(device)

# --- Training Loop with Validation Tracking ---
def train_scratch_model(epochs=5):
    weight_file = "scratch_weights.pt"

    if os.path.exists(weight_file):
        print(f"\nLoading checkpoint '{weight_file}'...")
        state_dict = torch.load(weight_file, map_location=device)
        try: model.load_state_dict(state_dict)
        except RuntimeError:
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "_orig_mod.")
            model.load_state_dict(state_dict)
        print("Model loaded. Skipping training.\n")
        return

    print(f"\nTraining Model... Tracking progress via Train Loss vs Validation Loss!")
    train_dataset = ScratchTokenDataset(train_tokens, block_size=BLOCK_SIZE)
    val_dataset = ScratchTokenDataset(val_tokens, block_size=BLOCK_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch_y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()

        # Quality Check against Validation Set
        model.eval()
        total_val_loss = 0
        with torch.inference_mode():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_logits = model(val_x)
                val_loss = F.cross_entropy(val_logits.view(-1, VOCAB_SIZE), val_y.view(-1))
                total_val_loss += val_loss.item()
        
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f" Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    unwrapped_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(unwrapped_model.state_dict(), weight_file)
    print(f"Training complete. Weights saved to '{weight_file}'.\n")

# --- Top-P Sampling Filtering ---
def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

# --- Dynamic Real-Time Text Generation Streamer ---
def run_scratch_generation(prompt, max_new_tokens=40, temperature=0.7, top_p=0.85, repetition_penalty=1.3):
    model.eval()
    
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
    generated_tokens = []

    print(f"--- PROMPT ---\n{prompt}\n\n--- DEEP MULTI-HEAD TRANSFORMER RESPONSE ---")
    start_time = time.time()

    # Tracks decoded character limits to avoid double-printing merged text fragments
    printed_len = len(prompt)
    print(prompt, end="", flush=True)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            input_crop = input_ids[:, -BLOCK_SIZE:]
            logits = model(input_crop, retain_attention=False)
            next_token_logits = logits[0, -1, :].clone()

            for token_id in set(generated_tokens[-15:]):
                if next_token_logits[token_id] > 0: next_token_logits[token_id] /= repetition_penalty
                else: next_token_logits[token_id] *= repetition_penalty

            next_token_logits = next_token_logits / temperature
            next_token_logits = top_p_filtering(next_token_logits, top_p=top_p)
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            token_val = next_token_id.item()

            if token_val == EOS_IDX:
                break

            generated_tokens.append(token_val)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            # Decode the cumulative stream to let byte formatting resolve natively
            full_decoded = tokenizer.decode(input_ids[0].tolist())
            next_chars = full_decoded[printed_len:]
            print(next_chars, end="", flush=True)
            printed_len = len(full_decoded)

        # Map attention paths for the updated generation context window
        _ = model(input_ids[:, -BLOCK_SIZE:], retain_attention=True)

    total_time = time.time() - start_time
    print(f"\n\n[Perf Stat: {len(generated_tokens) / total_time:.2f} tokens/sec]")
    
    return tokenizer.decode(input_ids[0].tolist()).split()

# --- Matplotlib Attention Visualization Engine ---
def plot_attention_map(sequence_words):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    target_block = model._orig_mod.blocks[-1] if hasattr(model, "_orig_mod") else model.blocks[-1]
    if target_block.attention.last_attention_weights is None: return

    attn_matrix = target_block.attention.last_attention_weights.numpy()
    seq_len = attn_matrix.shape[0]
    display_words = sequence_words[-seq_len:]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attn_matrix, cmap="plasma")
    
    ax.set_xticks(np.arange(len(display_words)))
    ax.set_yticks(np.arange(len(display_words)))
    ax.set_xticklabels(display_words, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(display_words, fontsize=8)
    
    ax.set_title("Transformer Context Attention Map Sub-Layer")
    plt.colorbar(im, label="Softmax Attention Magnitude")
    fig.tight_layout()
    
    output_img = "attention_map.png"
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"[Visualizer] Attention map saved as: '{os.path.abspath(output_img)}'")

if __name__ == "__main__":
    train_scratch_model(epochs=5)

    prompts = [
        "human decision-making is fundamentally",
        "the structural mechanics of a stellar core"
    ]

    for prompt in prompts:
        sentence_words = run_scratch_generation(prompt)
        if sentence_words:
            plot_attention_map(sentence_words)
        print("\n" + "="*60 + "\n")