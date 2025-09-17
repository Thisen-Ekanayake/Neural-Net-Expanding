import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
import json

# -------------------------
# Model classes (from training)
# -------------------------
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
    def __init__(self,cfg):
        super().__init__()
        V,S,D,L,M = cfg['vocab_size'], cfg['seq_len'], cfg['d_model'], cfg['n_layers'], cfg['mlp_dim']
        self.tok_emb = nn.Embedding(V,D)
        self.pos_emb = nn.Parameter(torch.zeros(1,S,D))
        self.blocks = nn.ModuleList([TransformerBlock(D,cfg['n_heads'],M) for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D,V,bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx):
        b,t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -------------------------
# Model config
# -------------------------
CONFIG = {
    'vocab_size': 4000,
    'seq_len': 512,
    'd_model': 256,
    'n_layers': 5,
    'mlp_dim': 1024,
    'n_heads': 8,
}

# -------------------------
# Load checkpoint
# -------------------------
checkpoint_path = "checkpoint/final.pt"
ckpt = torch.load(checkpoint_path, map_location='cpu')
model_state = ckpt['model_state']

model = TinyTransformerLM(CONFIG)
model.load_state_dict(model_state)
model.eval()

# -------------------------
# Load SentencePiece tokenizer
# -------------------------
tokenizer_path = "unigram_4000_0.9995.model"
sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

def tokenize(text):
    return sp.encode(text, out_type=int)

def detokenize(token_ids):
    return sp.decode(token_ids)

# -------------------------
# Greedy generation
# -------------------------
def generate(prompt, max_length=50):
    token_ids = tokenize(prompt)
    input_tensor = torch.tensor([token_ids])
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_tensor)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
    return detokenize(input_tensor[0].tolist())

# -------------------------
# Example
# -------------------------
prompt = ""
output = generate(prompt, max_length=50)
print("Generated text:", output)

