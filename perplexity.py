# perplexity_jsonl.py
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "checkpoint/final.pt"
DATASET_PATH = "tokenized_yt_dataset.jsonl"  # JSONL, each line: {"input_ids": [...]}
BATCH_SIZE = 16
SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN = 0  # assuming 0 is your padding token

# -----------------------------
# Dataset
# -----------------------------
class TokenizedJSONLDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                ids = item['input_ids']
                if len(ids) < 2:
                    continue
                self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        # Input = all except last token
        x = torch.tensor(ids[:-1], dtype=torch.long)
        # Target = all except first token
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

# -----------------------------
# Load model
# -----------------------------
class TinyTransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=256, n_layers=5, mlp_dim=1024, n_heads=8):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = torch.nn.ModuleList([TransformerBlock(d_model, n_heads, mlp_dim) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)
        if self.head.weight.shape == self.tok_emb.weight.shape:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Reuse your transformer block definitions
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, mlp_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, mlp_dim)
        self.fc2 = torch.nn.Linear(mlp_dim, d_model)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3*d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, self.n_heads, 3*self.head_dim).permute(2,0,1,3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q / (self.head_dim ** 0.5)
        att = torch.einsum('h b i d, h b j d -> h b i j', q, k)
        att = torch.softmax(att, dim=-1)
        out = torch.einsum('h b i j, h b j d -> h b i d', att, v)
        out = out.permute(1,2,0,3).reshape(B,T,C)
        return self.proj(out)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, mlp_dim):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, mlp_dim)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# -----------------------------
# Perplexity calculation
# -----------------------------
def calculate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            # Flatten
            logits = logits.view(-1, logits.size(-1))
            yb = yb.view(-1)
            # compute cross-entropy ignoring PAD_TOKEN
            loss = F.cross_entropy(logits, yb, ignore_index=PAD_TOKEN, reduction='sum')
            total_loss += loss.item()
            total_tokens += (yb != PAD_TOKEN).sum().item()
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load dataset
    dataset = TokenizedJSONLDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    # You might need to adjust these hyperparameters to match your training config
    VOCAB_SIZE = ckpt['cfg']['vocab_size'] if 'cfg' in ckpt else 4000
    SEQ_LEN = ckpt['cfg']['seq_len'] if 'cfg' in ckpt else 512
    model = TinyTransformerLM(VOCAB_SIZE, SEQ_LEN)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)

    # Calculate perplexity
    avg_loss, ppl = calculate_perplexity(model, dataloader)
    print(f"Average cross-entropy loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")
