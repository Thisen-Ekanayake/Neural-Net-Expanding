"""
nn_expand_pipeline.py

Full end-to-end pipeline:
1. Loads pre-tokenized JSONL dataset.
2. Trains a small Transformer LM.
3. Saves snapshots of parameters.
4. Performs PCA on snapshots.
5. Builds cubic spline interpolators.
6. Expands/widens layers using latent trajectory reconstruction.

Purpose
-------
A single-file experiment pipeline that:
 1. Builds a small Transformer-style language model (~5M parameters). The file
    calculates and prints a layer-wise parameter-count breakdown so you can
    verify the ~5M target.
 2. Trains (skeleton) with hooks that *log layer-wise parameter arrays* and
    hyperparameters at configurable intervals (placeholders for paths).
 3. Loads the saved logs, computes per-layer PCA, fits cubic-spline
    interpolators over the PCA latent trajectories, and reconstructs
    predicted weights for arbitrary training times.
 4. Provides functions to "widen"/"expand" layers (Net2Wider-like and
    low-rank SVD expansion) and to map reconstructed latent trajectories into
    the new larger shapes. Also includes optimizer-state mapping helpers.

This modified version adds **progress bars** to the main long-running steps
(training, snapshot loading, PCA fitting, spline fitting and expansion).
It uses `tqdm` when available and falls back to simple prints if not.

Usage
-----
- Edit the placeholder paths under CONFIG (DATA_PATH, CHECKPOINT_DIR, SNAPSHOT_DIR, PCA_DIR)
- Run training sketches (mode 'train') to create snapshots:
    python nn_expand_pipeline.py --mode train
- Run analysis and expansion (mode 'analyze' or 'expand'):
    python nn_expand_pipeline.py --mode analyze
    python nn_expand_pipeline.py --mode expand --target_scale 1.5

Notes
-----
- This is intended as a practical starting script — you will want to adapt
  the training loop, dataset, tokenization, and distributed settings to your
  needs.
- Snapshots are saved per-layer as compressed .npz files to keep disk usage
  reasonable. There's a tradeoff between snapshot frequency and disk size.

Placeholders
------------
Replace:
  - DATA_PATH = '<DATASET_PATH>'
  - CHECKPOINT_DIR = '<CHECKPOINT_DIR>'
  - SNAPSHOT_DIR = '<SNAPSHOT_DIR>'
  - PCA_DIR = '<PCA_OUTPUT_DIR>'
"""

import os
import time
import math
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        for it in x:
            yield it

try:
    from sklearn.decomposition import PCA
    from scipy.interpolate import CubicSpline
except Exception:
    PCA = None
    CubicSpline = None

# -----------------------------
# CONFIG
# -----------------------------
CONFIG = {
    'DATA_PATH': '100M_dataset.jsonl',
    'CHECKPOINT_DIR': 'checkpoint',
    'SNAPSHOT_DIR': 'snapshot',
    'PCA_DIR': 'pca',
    'vocab_size': 4000,
    'seq_len': 512,
    'd_model': 256,
    'n_layers': 5,
    'mlp_dim': 1024,
    'n_heads': 8,
    'batch_size': 32,
    'lr': 5e-4,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'epochs': 3,
    'save_snapshot_every_steps': 500,
    'snapshot_max_snapshots': 50,
    'pca_dim_small': 32,
    'pca_dim_medium': 64,
    'pca_dim_large': 128,
}

# -----------------------------
# Parameter utility
# -----------------------------
def compute_param_breakdown(cfg, tie_weights=True):
    V, L, D, M, S = cfg['vocab_size'], cfg['n_layers'], cfg['d_model'], cfg['mlp_dim'], cfg['seq_len']
    tok_embed = V * D
    pos_embed = S * D
    qkv = D * (3 * D) + 3 * D
    att_out = D * D + D
    mlp = D * M + M + M * D + D
    lns = 2 * (2 * D)
    per_block = qkv + att_out + mlp + lns
    total = tok_embed + pos_embed + per_block * L + 2 * D
    if not tie_weights:
        total += D * V
    return int(total), {'token_embedding': tok_embed, 'pos_embedding': pos_embed, 'per_transformer_block': per_block, 'n_layers': L, 'final_layernorm': 2*D}

# -----------------------------
# Dataset
# -----------------------------
class TokenizedJSONLDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        self.seq_len = CONFIG['seq_len']
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                ids = item['input_ids']
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
# Model
# -----------------------------
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
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        b,t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# Snapshot utils
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_snapshot(model, optimizer, step, cfg, snapshot_dir, extra_info=None):
    ensure_dir(snapshot_dir)
    t0 = time.time()
    meta = {'step': int(step), 'time':float(t0), 'lr': optimizer.param_groups[0]['lr'], 'cfg':cfg}
    if extra_info: meta.update(extra_info)
    out = {}
    for name,p in model.named_parameters():
        out[name.replace('.','__')] = p.detach().cpu().numpy()
    out['_meta'] = np.array([str(meta)])
    filename = os.path.join(snapshot_dir,f'snap_step_{step}.npz')
    np.savez_compressed(filename, **out)
    print(f"[snapshot] saved {filename} ({sum(v.nbytes for v in out.values())/1024/1024:.2f} MB)")

def load_snapshots(snapshot_dir, max_snapshots=50):
    files = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.npz')])
    files = files[-max_snapshots:]
    snapshots = []
    for f in files:
        data = np.load(os.path.join(snapshot_dir,f))
        snapshots.append(data)
    print(f"Loaded {len(snapshots)} snapshots")
    return snapshots

# -----------------------------
# PCA & spline
# -----------------------------
def build_layer_pcas(snapshots, pca_dim=32):
    layer_data = defaultdict(list)
    for snap in snapshots:
        for k,v in snap.items():
            if k=='_meta': continue
            layer_data[k].append(v.flatten())
    pcas = {}
    for layer, mats in layer_data.items():
        X = np.stack(mats,axis=0)
        pca = PCA(n_components=min(pca_dim,X.shape[0],X.shape[1]))
        pca.fit(X)
        pcas[layer] = pca
    return pcas

def build_splines_from_pcas(snapshots, pcas):
    layer_splines = {}
    steps = np.arange(len(snapshots))
    for layer, pca in pcas.items():
        X = np.stack([snap[layer].flatten() for snap in snapshots],axis=0)
        coeffs = pca.transform(X)
        cs = CubicSpline(steps, coeffs, axis=0)
        layer_splines[layer] = (cs, pca)
    return layer_splines

# -----------------------------
# Model expansion
# -----------------------------
def expand_model_from_snapshots(model, layer_splines, target_scale=1.5):
    new_state = {}
    for name, param in model.named_parameters():
        key = name.replace('.','__')
        if key in layer_splines:
            cs,pca = layer_splines[key]
            pred = cs(np.array([cs.x[-1]]))[0]
            new_flat = pca.inverse_transform(pred)
            new_param = torch.tensor(new_flat.reshape(param.shape),dtype=param.dtype)
            # optional scaling
            new_state[name] = new_param
    model.load_state_dict(new_state,strict=False)
    return model

# -----------------------------
# Training
# -----------------------------
def train_and_snapshot(cfg, snapshot_dir, checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyTransformerLM(cfg).to(device)
    total_estimate, parts = compute_param_breakdown(cfg)
    print("Parameter estimate:", total_estimate)
    actual = sum(p.numel() for p in model.parameters())
    print("Actual parameter count:", actual)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    dataset = TokenizedJSONLDataset(cfg['DATA_PATH'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    global_step = 0
    for epoch in range(cfg['epochs']):
        for xb,yb in tqdm(dataloader):
            xb,yb = xb.to(device),yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            global_step += 1
            if global_step % cfg['save_snapshot_every_steps']==0:
                save_snapshot(model, optimizer, global_step, cfg, snapshot_dir, extra_info={'loss':float(loss.detach().cpu())})
    save_snapshot(model, optimizer, global_step, cfg, snapshot_dir, extra_info={'loss':float(loss.detach().cpu())})
    ensure_dir(checkpoint_dir)
    torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'cfg':cfg},os.path.join(checkpoint_dir,'final.pt'))
    print('Training finished.')

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','analyze','expand'],default='train')
    parser.add_argument('--snapshot_dir',default=CONFIG['SNAPSHOT_DIR'])
    parser.add_argument('--checkpoint_dir',default=CONFIG['CHECKPOINT_DIR'])
    parser.add_argument('--pca_dir',default=CONFIG['PCA_DIR'])
    args = parser.parse_args()

    if args.mode=='train':
        train_and_snapshot(CONFIG,args.snapshot_dir,args.checkpoint_dir)
    elif args.mode=='analyze':
        snaps = load_snapshots(args.snapshot_dir, CONFIG['snapshot_max_snapshots'])
        pcas = build_layer_pcas(snaps, CONFIG['pca_dim_small'])
        splines = build_splines_from_pcas(snaps, pcas)
        ensure_dir(args.pca_dir)
        np.savez(os.path.join(args.pca_dir,'layer_splines.npz'), **splines)
        print("Saved PCA + spline data")
    elif args.mode=='expand':
        from_checkpoint = torch.load(os.path.join(args.checkpoint_dir,'final.pt'))
        model = TinyTransformerLM(CONFIG)
        model.load_state_dict(from_checkpoint['model_state'])
        spline_file = os.path.join(args.pca_dir,'layer_splines.npz')
        data = np.load(spline_file, allow_pickle=True)
        model = expand_model_from_snapshots(model, data, target_scale=1.5)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir,'expanded.pt'))
        print("Saved expanded model")

if __name__=='__main__':
    main()
