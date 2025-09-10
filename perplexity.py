import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
from tqdm.auto import tqdm

# -----------------------------
# Config (update paths)
# -----------------------------
PT_MODEL_PATH = "checkpoint/final.pt"
TOKENIZED_JSONL = "100M_dataset.jsonl"
BATCH_SIZE = 32
SEQ_LEN = 512
VOCAB_SIZE = 4000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Dataset
# -----------------------------
class TokenizedJSONLDataset(Dataset):
    def __init__(self, jsonl_path, seq_len=512):
        self.data = []
        self.seq_len = seq_len
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                ids = json.loads(line)['input_ids']
                if len(ids) != self.seq_len:
                    ids = ids[:self.seq_len] + [0]*(self.seq_len - len(ids))
                self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.roll(x, -1)
        return x, y

# -----------------------------
# Model (must match training)
# -----------------------------
import math
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, d_model)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, self.n_heads, 3*self.head_dim)
        qkv = qkv.permute(2,0,1,3)
        q,k,v = torch.chunk(qkv,3,dim=-1)
        q = q / math.sqrt(self.head_dim)
        att = torch.einsum('h b i d, h b j d -> h b i j', q, k)
        att = torch.softmax(att, dim=-1)
        out = torch.einsum('h b i j, h b j d -> h b i d', att, v)
        out = out.permute(1,2,0,3).reshape(B,T,C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model,n_heads)
        self.ff = FeedForward(d_model,mlp_dim)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=5, mlp_dim=1024, n_heads=8, seq_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1,seq_len,d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model,n_heads,mlp_dim) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model,vocab_size,bias=False)
        self.head.weight = self.tok_emb.weight
    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb[:, :idx.size(1), :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# Load model
# -----------------------------
checkpoint = torch.load(PT_MODEL_PATH, map_location=DEVICE)
cfg = checkpoint.get('cfg', {})
model = TinyTransformerLM(vocab_size=VOCAB_SIZE).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# -----------------------------
# Load dataset
# -----------------------------
dataset = TokenizedJSONLDataset(TOKENIZED_JSONL, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Compute Perplexity
# -----------------------------
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for xb, yb in tqdm(loader):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += yb.numel()

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)

print(f"Avg cross-entropy loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
