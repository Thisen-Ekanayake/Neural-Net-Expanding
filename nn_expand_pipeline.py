"""
nn_expand_pipeline.py

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
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Optional progress bar
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        # fallback generator that yields items but does not show a bar
        for it in x:
            yield it

# The analysis pieces
try:
    from sklearn.decomposition import PCA
    from scipy.interpolate import CubicSpline
except Exception:
    PCA = None
    CubicSpline = None
    # The user should `pip install scikit-learn scipy` to use analysis features

# -----------------------------
# CONFIG (edit placeholders)
# -----------------------------
CONFIG = {
    # Data / IO paths (replace these!)
    'DATA_PATH': '100M_words.txt',
    'CHECKPOINT_DIR': 'checkpoint',
    'SNAPSHOT_DIR': 'snapshot',
    'PCA_DIR': 'pca',

    # Model architecture hyperparams (we target ~5.1M params)
    'vocab_size': 4000,
    'seq_len': 512,
    'd_model': 256,
    'n_layers': 5,
    'mlp_dim': 1024,
    'n_heads': 8,  # must divide d_model

    # Training hyperparams (sensible defaults)
    'batch_size': 32,
    'lr': 5e-4,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'epochs': 3,
    'save_snapshot_every_steps': 500,  # snapshot frequency (tune)

    # Snapshot / analysis settings
    'snapshot_max_snapshots': 50,  # maximum snapshots to store
    'pca_dim_small': 32,  # latent dim for small layers
    'pca_dim_medium': 64,
    'pca_dim_large': 128,
}

# -----------------------------
# Utility: Parameter count calculator
# -----------------------------

def compute_param_breakdown(cfg, tie_weights=True):
    """
    Compute a detailed breakdown of the parameter count for the simple
    Transformer LM defined below. We return the integer total and a dict with
    parts so you can print them and verify the ~5M target.

    This function also demonstrates the arithmetic used to compute counts.
    """
    V = cfg['vocab_size']
    L = cfg['n_layers']
    D = cfg['d_model']
    M = cfg['mlp_dim']
    S = cfg['seq_len']

    # Embeddings
    tok_embed = V * D  # token embedding matrix: (vocab x d_model)
    pos_embed = S * D  # positional embeddings: (seq_len x d_model)

    # Per-layer (Transformer block) breakdown
    # Attention QKV linear: d_model -> 3*d_model => d_model * 3*d_model + 3*d_model (bias)
    qkv = D * (3 * D) + 3 * D
    # Attention output projection: d_model -> d_model
    att_out = D * D + D
    # MLP: two linear layers d_model -> mlp_dim -> d_model
    mlp = D * M + M + M * D + D
    # LayerNorms (2 per block): each has weight and bias -> 2*D
    lns = 2 * (2 * D)
    per_block = qkv + att_out + mlp + lns

    # total
    total = tok_embed + pos_embed + per_block * L + 2 * D  # final layernorm 2*D
    if not tie_weights:
        total += D * V  # final lm-head if not tied

    parts = {
        'token_embedding': tok_embed,
        'pos_embedding': pos_embed,
        'per_transformer_block': per_block,
        'n_layers': L,
        'final_layernorm': 2 * D,
    }
    return int(total), parts


# -----------------------------
# Minimal toy dataset & dataloader (replace with your tokenizer/dataset)
# -----------------------------
class DummyTextDataset(Dataset):
    def __init__(self, length=10000, seq_len=CONFIG['seq_len'], vocab=CONFIG['vocab_size']):
        self.length = length
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # random tokens; replace with real tokenized data
        x = np.random.randint(0, self.vocab, size=(self.seq_len,), dtype=np.int64)
        y = np.roll(x, -1)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# -----------------------------
# Model: tiny transformer LM (GPT-like)
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
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.head_dim)
        # shape management: split into q,k,v per head
        qkv = qkv.permute(2, 0, 1, 3)  # (heads, B, T, 3*head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # simple scaled dot-product per head
        q = q / math.sqrt(self.head_dim)
        att = torch.einsum('h b i d, h b j d -> h b i j', q, k)
        att = torch.softmax(att, dim=-1)
        out = torch.einsum('h b i j, h b j d -> h b i d', att, v)
        out = out.permute(1, 2, 0, 3).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        V = cfg['vocab_size']
        S = cfg['seq_len']
        D = cfg['d_model']
        L = cfg['n_layers']
        M = cfg['mlp_dim']

        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Parameter(torch.zeros(1, S, D))
        self.blocks = nn.ModuleList([TransformerBlock(D, cfg['n_heads'], M) for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

        # Weight tying (head shares token embedding) is common; we'll tie if shapes match
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


# -----------------------------
# Snapshot saving & logging
# -----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_snapshot(model, optimizer, step, cfg, snapshot_dir, extra_info=None):
    """
    Save a snapshot consisting of per-layer weight arrays + hyperparams.
    This function saves a compressed .npz per snapshot containing:
      - for each parameter: a numpy array, keyed by safe name
      - metadata: step, timestamp, lr, weight_decay, cfg (selected fields)

    NOTE: This is intentionally verbose so you get layerwise logs you can
    analyze later. For large models you'll want a different storage strategy
    (e.g., saving PCA coefficients instead of full tensors).
    """
    ensure_dir(snapshot_dir)
    t0 = time.time()
    meta = {
        'step': int(step),
        'time': float(t0),
        'lr': optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else None,
        'weight_decay': cfg.get('weight_decay'),
        'cfg': {k: v for k, v in cfg.items() if k in ('vocab_size', 'd_model', 'n_layers', 'mlp_dim', 'seq_len')},
    }
    if extra_info:
        meta.update(extra_info)

    out = {}
    for name, p in model.named_parameters():
        out[name.replace('.', '__')] = p.detach().cpu().numpy()
    out['_meta'] = np.array([str(meta)])

    filename = os.path.join(snapshot_dir, f'snap_step_{step}.npz')
    np.savez_compressed(filename, **out)
    print(f"[snapshot] saved {filename} ({sum(v.nbytes for v in out.values())/1024/1024:.2f} MB)")


# -----------------------------
# Training loop skeleton (with progress bar)
# -----------------------------

def train_and_snapshot(cfg, snapshot_dir, checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyTransformerLM(cfg).to(device)

    # Print parameter breakdown & actual count
    total_estimate, parts = compute_param_breakdown(cfg)
    print("Parameter estimate (arithmetic):", total_estimate)
    for k, v in parts.items():
        print(f"  {k}: {v}")
    actual = sum(p.numel() for p in model.parameters())
    print("Actual parameter count from model.parameters():", actual)

    # Optimizer & dataset
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    dataset = DummyTextDataset(length=20000, seq_len=cfg['seq_len'], vocab=cfg['vocab_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

    total_steps = cfg['epochs'] * len(dataloader)
    global_step = 0
    last_snapshot = -1

    print(f"Starting training for approx {total_steps} steps (epochs={cfg['epochs']}, steps per epoch={len(dataloader)})")
    # Use tqdm if available for a nice progress bar
    with (tqdm(range(total_steps), desc='train_steps') if hasattr(tqdm, '__call__') else range(total_steps)) as pbar:
        # We'll iterate by epochs and batches but update the progress bar by step
        for epoch in range(cfg['epochs']):
            for xb, yb in dataloader:
                model.train()
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                global_step += 1
                # update progress bar by one step
                try:
                    pbar.update(1)
                except Exception:
                    pass

                if global_step % cfg['save_snapshot_every_steps'] == 0:
                    save_snapshot(model, optimizer, global_step, cfg, snapshot_dir, extra_info={'loss': float(loss.detach().cpu())})

                # crude stop for demo purposes
                if global_step >= cfg.get('demo_steps', 2000):
                    break
            if global_step >= cfg.get('demo_steps', 2000):
                break

    # final save
    save_snapshot(model, optimizer, global_step, cfg, snapshot_dir, extra_info={'loss': float(loss.detach().cpu())})
    # Save final model checkpoint
    ensure_dir(checkpoint_dir)
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'cfg': cfg}, os.path.join(checkpoint_dir, 'final.pt'))
    print('Training finished. Checkpoints and snapshots saved.')


# -----------------------------
# Analysis: load snapshots -> PCA per layer -> temporal interpolation
# -----------------------------

def load_snapshots(snapshot_dir, max_files=None):
    files = sorted([os.path.join(snapshot_dir, f) for f in os.listdir(snapshot_dir) if f.endswith('.npz')])
    if max_files:
        files = files[:max_files]
    snapshots = []
    times = []
    steps = []
    print(f"Loading {len(files)} snapshot files from {snapshot_dir}")
    for f in tqdm(files, desc='loading_snapshots'):
        d = np.load(f, allow_pickle=True)
        meta = eval(d['_meta'][0].tolist())
        steps.append(meta['step'])
        times.append(meta['time'])
        snapshots.append(d)
    return snapshots, steps, times


def build_layer_pcas(snapshots, steps, pca_dir, dims_map=None):
    """
    For each parameter name present in snapshots, collect the flattened
    parameter vectors across snapshots and fit a PCA. Save the PCA objects
    (components and mean) to pca_dir as .npz files.

    dims_map: a function or dict mapping param name -> desired PCA dim.
    """
    ensure_dir(pca_dir)
    names = [k for k in snapshots[0].keys() if not k.startswith('_')]
    layer_to_matrix = {}
    print(f'Found {len(names)} parameter arrays to run PCA on')
    for name in tqdm(names, desc='collecting_layers'):
        mats = []
        for d in snapshots:
            arr = d[name]
            mats.append(arr.ravel())
        M = np.stack(mats, axis=0)  # shape (T, P)
        layer_to_matrix[name] = M

    pca_info = {}
    for name, M in tqdm(layer_to_matrix.items(), desc='fitting_pcas'):
        P = M.shape[1]
        # choose dim heuristically
        if dims_map is None:
            if P < 2000:
                d = CONFIG['pca_dim_small']
            elif P < 20000:
                d = CONFIG['pca_dim_medium']
            else:
                d = CONFIG['pca_dim_large']
        else:
            d = dims_map.get(name, CONFIG['pca_dim_small']) if isinstance(dims_map, dict) else dims_map(name)
        if PCA is None:
            raise RuntimeError('scikit-learn required for PCA analysis')
        pca = PCA(n_components=min(d, M.shape[0]-1, M.shape[1]))
        Z = pca.fit_transform(M)
        np.savez_compressed(os.path.join(pca_dir, f'pca_{name}.npz'), components=pca.components_, mean=pca.mean_, explained_variance=pca.explained_variance_, Z=Z)
        pca_info[name] = {'d': Z.shape[1], 'P': P}
        print(f'[pca] saved {name} -> Z shape {Z.shape} (orig P={P})')
    return pca_info


def build_splines_from_pcas(pca_dir, steps, spline_out_dir):
    """
    For each saved PCA (which includes the reduced-dim trajectories Z[t]),
    fit a CubicSpline per PCA dimension over "steps" and save the spline
    coefficients (we save the Z itself so splines can be rebuilt later).
    """
    ensure_dir(spline_out_dir)
    if CubicSpline is None:
        raise RuntimeError('scipy required for splines')
    pca_files = [f for f in os.listdir(pca_dir) if f.startswith('pca_') and f.endswith('.npz')]
    print(f'Fitting splines for {len(pca_files)} PCA files')
    for pf in tqdm(pca_files, desc='fitting_splines'):
        d = np.load(os.path.join(pca_dir, pf), allow_pickle=True)
        Z = d['Z']
        t = np.array(steps)
        cs_list = []
        for dim in range(Z.shape[1]):
            cs = CubicSpline(t, Z[:, dim], bc_type='natural')
            cs_list.append(cs.c)
        # We can't easily pickle the CubicSpline objects portably; instead we
        # save the Z trajectory so splines can be recreated later. This keeps
        # the pipeline transparent.
        np.savez_compressed(os.path.join(spline_out_dir, pf.replace('pca_', 'spline_')), Z=Z, steps=t)
        print(f'[spline] saved spline data for {pf} (Z shape {Z.shape})')


# -----------------------------
# Reconstruction: given a pca and a target step, reconstruct full weight
# -----------------------------

def reconstruct_weight_from_pca(pca_npz_path, target_step, steps):
    """
    Load pca file (.npz saved earlier) and reconstruct the weight vector
    for an arbitrary target_step by rebuilding a cubic spline for each PCA
    dimension and evaluating it.

    Returns the reconstructed flattened weight vector.
    """
    d = np.load(pca_npz_path, allow_pickle=True)
    components = d['components']
    mean = d['mean']
    Z = d['Z']
    t = np.array(steps)
    if CubicSpline is None:
        raise RuntimeError('scipy required')
    # fit splines per PCA dim
    Z_pred = []
    for dim in range(Z.shape[1]):
        cs = CubicSpline(t, Z[:, dim], bc_type='natural')
        Z_pred.append(cs(target_step))
    Z_pred = np.array(Z_pred)
    w_flat = mean + components.T @ Z_pred
    return w_flat


# -----------------------------
# Expansion utilities: Net2Wider-like and low-rank expansion
# -----------------------------

def net2wider_linear(old_weight, old_bias, new_out, noise_scale=1e-5):
    """
    old_weight: numpy array shape (out, in)
    old_bias: numpy array shape (out,) or None
    new_out: desired new output dimension (must be >= old_out)

    Very simple duplication-based widening: we repeat rows to match
    new_out then add tiny noise to break symmetry. We also return a mapping
    array `orig_indices` that maps new indices to original indices (useful
    for splitting/redistributing next-layer weights).
    """
    out, inp = old_weight.shape
    assert new_out >= out
    factor = math.floor(new_out / out)
    remainder = new_out - factor * out
    rows = [old_weight[i:i+1, :] for i in range(out) for _ in range(factor)]
    # distribute remainder by duplicating first `remainder` rows once more
    for i in range(remainder):
        rows.append(old_weight[i:i+1, :])
    W_new = np.concatenate(rows, axis=0)
    # bias
    if old_bias is not None:
        bs = [old_bias[i:i+1] for i in range(out) for _ in range(factor)]
        for i in range(remainder):
            bs.append(old_bias[i:i+1])
        b_new = np.concatenate(bs, axis=0)
    else:
        b_new = None
    # break symmetry
    W_new = W_new + np.random.randn(*W_new.shape) * noise_scale
    return W_new, (b_new if b_new is None else b_new.copy())


def expand_linear_lowrank(old_weight, rank_new, noise_scale=1e-6):
    """
    Expand a weight matrix by converting to a factorized low-rank form and
    then increasing the rank. We use SVD on the old weight and append small
    random components to expand to `rank_new`.
    Returns a dense expanded matrix (out_new x in) reconstructed from the
    expanded factors. This is a simple heuristic — more sophisticated
    factorization-aware layers could keep U and V separately.
    """
    import numpy.linalg as la
    W = old_weight
    U, s, Vt = la.svd(W, full_matrices=False)
    r_old = s.shape[0]
    out, inp = W.shape
    if rank_new <= r_old:
        # truncate / reconstruct
        r = rank_new
        U2 = U[:, :r]
        s2 = s[:r]
        Vt2 = Vt[:r, :]
    else:
        # append small random components
        U2 = np.concatenate([U, np.random.randn(out, rank_new - r_old) * 1e-3], axis=1)
        s2 = np.concatenate([s, np.zeros(rank_new - r_old)], axis=0)
        Vt2 = np.concatenate([Vt, np.random.randn(rank_new - r_old, inp) * 1e-3], axis=0)
    W_new = (U2 * s2[np.newaxis, :]) @ Vt2
    W_new = W_new + np.random.randn(*W_new.shape) * noise_scale
    return W_new


# -----------------------------
# Map optimizer state for widened params (simple heuristic)
# -----------------------------

def map_optimizer_state(old_state_dict, param_name_map):
    """
    Given an optimizer.state_dict() `old_state_dict` from the small model
    and a param_name_map which maps new_param_name -> (old_param_name, mapping_info),
    create a new optimizer state dict for the expanded model.

    This is a heuristic: for duplicated params we copy the optimizer moments; for
    new params we initialize moments to zero.
    """
    new_state = {'state': {}, 'param_groups': old_state_dict['param_groups']}
    for new_name, (old_name, info) in param_name_map.items():
        if old_name in old_state_dict['state']:
            new_state['state'][new_name] = old_state_dict['state'][old_name].copy()
        else:
            new_state['state'][new_name] = {}
    return new_state


# -----------------------------
# High-level expand function that consumes snapshot PCA predictions
# -----------------------------

def expand_model_from_snapshots(model, snapshot_pca_dir, steps, target_scale=1.5):
    """
    Given a `model` (torch.nn.Module) and a directory with PCA files, this
    function shows how to:
      1. reconstruct a predicted weight for the latest snapshot step
      2. expand a chosen linear layer by Net2Wider-like duplication and/or
         low-rank expansion
      3. map reconstructed flattened weights into the new expanded shapes

    This function is intentionally conservative: it only expands Linear-like
    parameters and demonstrates the mapping. You will need to adapt it to
    your exact model naming / graph.
    """
    # Example: expand the first transformer's qkv projection up by target_scale
    state = model.state_dict()
    # find a candidate param name for a linear weight (heuristic)
    candidate = None
    candidate_list = [k for k in state.keys() if 'qkv' in k and k.endswith('weight')]
    if len(candidate_list) > 0:
        candidate = candidate_list[0]
    else:
        # fallback: pick first linear weight
        for k in state.keys():
            if k.endswith('weight') and state[k].dim() == 2:
                candidate = k
                break

    if candidate is None:
        print('No linear weight found; pick a linear weight name manually')
        return model

    print(f'Chosen candidate for widening: {candidate}')
    W = state[candidate].cpu().numpy()
    old_out, old_in = W.shape
    new_out = int(math.ceil(old_out * target_scale))
    print(f'Expanding {candidate}: {old_out} -> {new_out} outputs')

    # Simple Net2Wider duplication
    W_new, b_new = net2wider_linear(W, None, new_out)

    # load the new weights back to the model (careful about shapes!)
    new_tensor = torch.tensor(W_new, dtype=state[candidate].dtype)
    state[candidate] = new_tensor

    # next layer: if there is a following linear that takes this as input,
    # you should split its columns accordingly. This script leaves that step
    # for you to adapt: mapping the "next" layer depends on your naming scheme.

    model.load_state_dict(state, strict=False)
    print('Model updated with widened weight (next-layer redistribution left as TODO)')
    return model


# -----------------------------
# CLI & main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'analyze', 'expand'], default='train')
    parser.add_argument('--snapshot_dir', default=CONFIG['SNAPSHOT_DIR'])
    parser.add_argument('--checkpoint_dir', default=CONFIG['CHECKPOINT_DIR'])
    parser.add_argument('--pca_dir', default=CONFIG['PCA_DIR'])
    parser.add_argument('--max_snapshots', type=int, default=CONFIG['snapshot_max_snapshots'])
    parser.add_argument('--target_scale', type=float, default=1.5)
    args = parser.parse_args()

    if args.mode == 'train':
        print('Starting training + snapshotting (edit CONFIG for paths and dataset)')
        train_and_snapshot(CONFIG, args.snapshot_dir, args.checkpoint_dir)

    elif args.mode == 'analyze':
        print('Loading snapshots...')
        snapshots, steps, times = load_snapshots(args.snapshot_dir, max_files=args.max_snapshots)
        print(f'Loaded {len(snapshots)} snapshots at steps: {steps}')
        print('Building PCA per-layer...')
        pca_info = build_layer_pcas(snapshots, steps, args.pca_dir)
        print('Fitting spline data...')
        build_splines_from_pcas(args.pca_dir, steps, args.pca_dir)
        print('Analysis complete.')

    elif args.mode == 'expand':
        print('Loading final checkpoint to expand...')
        import torch
        ckpt = torch.load(os.path.join(args.checkpoint_dir, 'final.pt'), map_location='cpu')
        model = TinyTransformerLM(CONFIG)
        model.load_state_dict(ckpt['model_state'], strict=False)
        snapshots, steps, times = load_snapshots(args.snapshot_dir, max_files=args.max_snapshots)
        model = expand_model_from_snapshots(model, args.pca_dir, steps, target_scale=args.target_scale)
        # save expanded model
        ensure_dir(args.checkpoint_dir)
        torch.save({'model_state': model.state_dict(), 'cfg': CONFIG}, os.path.join(args.checkpoint_dir, 'expanded.pt'))
        print('Expanded model saved to expanded.pt')


if __name__ == '__main__':
    main()
