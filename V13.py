import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# --- Hyperparameters ---
batch_size = 64     # How many independent sequences to process in parallel
block_size = 64     # Maximum context length (tokens per sequence)
max_epochs = 10     # Total passes through the dataset
learning_rate = 1e-3
n_embd = 128        # Embedding dimension size
n_head = 4          # Multi-head attention heads
n_layer = 4         # Transformer blocks
dropout = 0.1

# Automatically target Apple Silicon GPU (mps), Nvidia (cuda), or fallback to CPU
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
# -----------------------

# 1. Load and Parse Dataset
if not os.path.exists('essays.txt'):
    raise FileNotFoundError("Could not find 'essays.txt'. Run your download setup first!")

with open('essays.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Setup character vocabulary mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert entire corpus to integer tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset size: {len(data):,} tokens | Vocab size: {vocab_size} unique chars")

# 2. V14 Non-Overlapping Strided Chunking Logic
num_chunks = (len(data) - 1) // block_size

# Slice data directly into clean, non-overlapping blocks
X_all = torch.stack([data[i*block_size : i*block_size + block_size] for i in range(num_chunks)])
Y_all = torch.stack([data[i*block_size + 1 : i*block_size + block_size + 1] for i in range(num_chunks)])

num_batches = num_chunks // batch_size
print(f"Total structured sequences: {num_chunks:,}")
print(f"Engineered batches per epoch: {num_batches} (No redundancy)")

# 3. Transformer Core Architecture Modules
class Head(nn.Module):
    """ One head of causal self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        wei = q @ k.transpose(-2, -1) * (C**-0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple attention heads running in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Standard MLP layer norm computation """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer layer: communication layer followed by computational logic """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# 4. Initialization
model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 5. Training Loop
print("\n--- Starting Training Engine ---")
for epoch in range(max_epochs):
    # Shuffle sequence order at the start of each epoch to break structural linearity
    rand_indices = torch.randperm(num_chunks)
    X_shuffled = X_all[rand_indices]
    Y_shuffled = Y_all[rand_indices]
    
    epoch_loss = 0.0
    model.train()
    
    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size
        end_i = start_i + batch_size
        
        xb = X_shuffled[start_i:end_i].to(device)
        yb = Y_shuffled[start_i:end_i].to(device)
        
        # Optimize step
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 100 == 0 or batch_idx == num_batches - 1:
            print(f"Epoch {epoch+1:02d}/{max_epochs:02d} | Batch {batch_idx:03d}/{num_batches} | Loss: {loss.item():.4f}")
            
    avg_loss = epoch_loss / num_batches
    print(f"\n>> Epoch {epoch+1} Complete | Avg Train Loss: {avg_loss:.4f}")
    
    # Live Vibe Check Generation Block
    model.eval()
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with newline placeholder
        sample_output = decode(model.generate(context, max_new_tokens=150)[0].tolist())
        print(f"--- Epoch {epoch+1} Sample Inference Output ---\n{sample_output}\n")
    print("="*60 + "\n")

print("Training finished successfully.")

# 6. Save the Weights Dictionary
weights_path = "v14_model_weights.pth"
torch.save(model.state_dict(), weights_path)
print(f"--> Success! Weights file saved to your workspace as: '{weights_path}'")