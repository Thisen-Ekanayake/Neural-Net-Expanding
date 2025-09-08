"""
param_logging_transformer_5M.py

Skeleton training pipeline that:
 - Builds a Transformer-style language model of ~5M parameters (configurable).
 - Shows a per-layer parameter-count breakdown and prints how the total was calculated.
 - Loads a text dataset and tokenizes it using a SentencePiece tokenizer (USER MUST SUPPLY PATH).
 - Creates a Dataset that packs tokens into 512-token blocks.
 - Trains for a small number of steps and logs detailed parameter/gradient/activation *statistics* to CSV.
 - Optionally dumps full-weight arrays (NPZ) every `full_dump_interval` steps (use sparingly).
 - Uses all CPU cores for tokenization & DataLoader workers.

USAGE:
 - Edit the placeholders near the top: `SP_MODEL_PATH`, `DATASET_TXT_PATH`, `CACHE_DIR`.
 - Run: python param_logging_transformer_5M.py

Notes about the tokenizer: you told me your tokenizer uses VOCAB_SIZE=4000, UNIGRAM model
and character coverage 0.9996. This script expects a SentencePiece model at SP_MODEL_PATH.

This script focuses on *statistics logging* (CSV) not full-parameter storage per step to avoid
astronomical storage use. If you want raw dumps, increase `full_dump_interval` and prepare
large disk space (NPZ files can be large).

"""

import os
import time
import math
import csv
import sys
import itertools
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- PLACEHOLDERS: edit these paths ---
SP_MODEL_PATH = "unigram_4000_0.9995.model"  # <-- your unigram tokenizer (vocab 4000)
DATASET_TXT_PATH = "100M_words.txt"  # <-- plain text file, one or many lines
CACHE_DIR = "nn_cache"  # <-- caching / checkpoints / dumps
# -------------------------------------

# TRAINING / MODEL HYPERPARAMS
VOCAB_SIZE = 4000
BLOCK_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How often to log/write CSV rows (in optimizer steps)
log_interval = 10
# How often to do a full raw-weights dump (NPZ). Keep large - set to 0 to disable.
full_dump_interval = 500

# Use all CPU cores for DataLoader and tokenization
NUM_WORKERS = os.cpu_count() or 1

os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------- Tokenizer loader (SentencePiece) -----------------
try:
    import sentencepiece as spm
except Exception as e:
    print("sentencepiece not installed. pip install sentencepiece")
    raise

sp = spm.SentencePieceProcessor()
if not os.path.exists(SP_MODEL_PATH):
    raise FileNotFoundError(f"SentencePiece model not found at {SP_MODEL_PATH}")
sp.Load(SP_MODEL_PATH)
assert sp.vocab_size() == VOCAB_SIZE, f"Tokenizer vocab {sp.vocab_size()} != expected {VOCAB_SIZE}"

# ----------------- Simple dataset that packs tokens into fixed-length blocks -----------------
class PackedTextDataset(Dataset):
    def __init__(self, txt_path, sp_processor, block_size=512, cache_dir=None):
        self.block_size = block_size
        self.sp = sp_processor
        self.cache_dir = cache_dir
        # We'll stream through the text and create token blocks.
        # For speed and memory, we do a single-pass tokenization and save token ids to a .pt cache.
        base_cache = os.path.join(cache_dir or ".", os.path.basename(txt_path) + f".sp_{block_size}.pt")
        if os.path.exists(base_cache):
            print(f"Loading token blocks from cache: {base_cache}")
            self.blocks = torch.load(base_cache)
        else:
            print("Tokenizing and packing dataset (may take time). Using all CPU cores for sentencepiece).")
            token_ids = []
            blocks = []
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = self.sp.EncodeAsIds(line)
                    token_ids.extend(ids)
                    # pack whenever possible
                    while len(token_ids) >= block_size:
                        blk = token_ids[:block_size]
                        blocks.append(torch.tensor(blk, dtype=torch.long))
                        token_ids = token_ids[block_size:]
            # drop remainder (or optionally pad)
            print(f"Created {len(blocks)} blocks from dataset.")
            torch.save(blocks, base_cache)
            self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        x = self.blocks[idx]
        # for language modeling we'll return input_ids and target_ids shifted by 1
        return x[:-1].long(), x[1:].long()

# ----------------- Small Transformer LM (GPT-like) -----------------
# We'll build a compact decoder-only stack using nn.TransformerEncoderLayer to keep code short.
class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layer=6, n_head=8, d_ff=None, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, d_model)
        d_ff = d_ff or (4 * d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # initialization similar to many lm scripts
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids):
        # input_ids: (B, T)
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        # transformer expects (T, B, C)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# --- Helper: print parameter counts module-wise and total (how we calculated params) ---
def print_param_breakdown(model):
    print("Parameter count breakdown (module -> numel):")
    breakdown = []
    for name, module in model.named_modules():
        # ignore top-level container
        if len(list(module.parameters(recurse=False))) == 0:
            continue
        module_params = 0
        for p in module.parameters(recurse=False):
            module_params += p.numel()
        if module_params > 0:
            breakdown.append((name or "<root>", module_params))
    total = sum(p.numel() for p in model.parameters())
    for name, cnt in breakdown:
        print(f"  {name:40s} : {cnt:,d}")
    print(f"Total parameters: {total:,d}")
    return total, breakdown

# ----------------- Build model to target ~5M params -----------------
# We'll pick d_model and n_layer that roughly give 5M; final truth is measured by summing params.
# Start with a compact setting and then adjust until it's about 5M.

# Experimentation values (you can tune these to reach ~5M):
# - d_model: 384 or 512
# - n_layer: 6..8
# - n_head: must divide d_model

# We'll try d_model=384, n_layer=6, n_head=8 as a starting point.
D_MODEL = 384
N_LAYER = 6
N_HEAD = 8
D_FF = 4 * D_MODEL

model = SimpleTransformerLM(VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF).to(DEVICE)

# Calculate parameters and print breakdown
total_params, breakdown = print_param_breakdown(model)
print(f"Model built with d_model={D_MODEL}, n_layer={N_LAYER}, n_head={N_HEAD}. Total params: {total_params:,d}")

# If total_params isn't ~5M, user can edit D_MODEL / N_LAYER. The definitive count is printed above.

# ----------------- Logging utilities -----------------
CSV_PATH = os.path.join(CACHE_DIR, "param_stats.csv")

csv_fieldnames = [
    "timestamp", "step", "epoch",
    "scope", "layer_name", "param_name", "param_type", "shape",
    "stat_type", "value"
]

# initialize CSV file with header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
        writer.writeheader()

# Helper: compute stats for a tensor (torch.Tensor)
def tensor_stats(tensor):
    # tensor may be on GPU; move to cpu for stat extraction but keep it in float
    t = tensor.detach().cpu()
    stats = {
        'mean': float(t.mean().item()),
        'std': float(t.std(unbiased=False).item()),
        'min': float(t.min().item()),
        'max': float(t.max().item()),
        'median': float(t.median().item()),
        'l2_norm': float(t.norm(2).item()),
        'l1_norm': float(t.norm(1).item()),
        'numel': int(t.numel()),
        'sparsity': float((t==0).sum().item()) / t.numel()
    }
    return stats

# Activation hooks: collect statistics per module name for the current forward pass
activation_stats = {}

def make_forward_hook(name):
    def hook(module, input, output):
        try:
            # handle tensor or tuple of tensors
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activation_stats[name] = tensor_stats(out.view(-1))
        except Exception as e:
            activation_stats[name] = {"error": str(e)}
    return hook

# register forward hooks on transformer layers (choose modules that produce activations)
for i, layer in enumerate(model.transformer.layers):
    # layer: TransformerEncoderLayer
    layer_name = f"transformer.layer.{i}"
    # hook on the self-attention output and feed-forward output (TransformerEncoderLayer has .linear1/.linear2)
    if hasattr(layer, 'linear1'):
        layer.linear1.register_forward_hook(make_forward_hook(layer_name + ".ffn_in"))
    if hasattr(layer, 'linear2'):
        layer.linear2.register_forward_hook(make_forward_hook(layer_name + ".ffn_out"))

# ----------------- Dataset and DataLoader -----------------
print(f"Using {NUM_WORKERS} DataLoader workers (os.cpu_count()).")
train_dataset = PackedTextDataset(DATASET_TXT_PATH, sp, block_size=BLOCK_SIZE, cache_dir=CACHE_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

# ----------------- Training setup -----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

global_step = 0
start_time = time.time()

# A little function to log stats for all parameters at a given step
def log_param_stats(step, epoch, extra_info=None):
    rows = []
    ts = time.time()
    # global metrics
    total_param_norm = 0.0
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        scope, _, param_name = name.rpartition('.')
        p_stats = tensor_stats(param.data)
        # gradients might be None
        if param.grad is not None:
            g_stats = tensor_stats(param.grad)
        else:
            g_stats = None

        # collect a few selected stats to keep CSV manageable
        selected = ['mean', 'std', 'min', 'max', 'l2_norm', 'sparsity']
        for stat in selected:
            rows.append({
                'timestamp': ts,
                'step': step,
                'epoch': epoch,
                'scope': scope,
                'layer_name': scope,
                'param_name': param_name,
                'param_type': 'param',
                'shape': str(list(param.shape)),
                'stat_type': stat,
                'value': p_stats[stat]
            })
            if g_stats is not None:
                rows.append({
                    'timestamp': ts,
                    'step': step,
                    'epoch': epoch,
                    'scope': scope,
                    'layer_name': scope,
                    'param_name': param_name,
                    'param_type': 'grad',
                    'shape': str(list(param.shape)),
                    'stat_type': stat,
                    'value': g_stats[stat]
                })
        total_param_norm += p_stats['l2_norm'] ** 2
        if g_stats is not None:
            total_grad_norm += g_stats['l2_norm'] ** 2

    total_param_norm = math.sqrt(total_param_norm)
    total_grad_norm = math.sqrt(total_grad_norm)
    # add global rows
    rows.append({
        'timestamp': ts,
        'step': step,
        'epoch': epoch,
        'scope': '<global>',
        'layer_name': '<global>',
        'param_name': '<global>',
        'param_type': 'global',
        'shape': '',
        'stat_type': 'total_param_l2_norm',
        'value': total_param_norm
    })
    rows.append({
        'timestamp': ts,
        'step': step,
        'epoch': epoch,
        'scope': '<global>',
        'layer_name': '<global>',
        'param_name': '<global>',
        'param_type': 'global',
        'shape': '',
        'stat_type': 'total_grad_l2_norm',
        'value': total_grad_norm
    })

    # activations
    for act_name, stats in activation_stats.items():
        for k, v in stats.items():
            rows.append({
                'timestamp': ts,
                'step': step,
                'epoch': epoch,
                'scope': act_name,
                'layer_name': act_name,
                'param_name': '<activation>',
                'param_type': 'activation',
                'shape': '',
                'stat_type': k,
                'value': v
            })

    # write rows to CSV
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
        for r in rows:
            writer.writerow(r)

    # optionally dump raw weights
    if full_dump_interval and step % full_dump_interval == 0:
        dump_path = os.path.join(CACHE_DIR, f"raw_weights_step_{step}.npz")
        print(f"Dumping raw weights to {dump_path}")
        dump_dict = {}
        for name, param in model.named_parameters():
            dump_dict[name.replace('.', '_')] = param.detach().cpu().numpy()
        # write NPZ (may be large)
        try:
            import numpy as np
            np.savez_compressed(dump_path, **dump_dict)
        except Exception as e:
            print("Failed to dump raw weights:", e)

# ----------------- Training loop (skeleton) -----------------
model.train()
for epoch in range(1, NUM_EPOCHS + 1):
    for batch in train_loader:
        input_ids, target_ids = batch
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)

        optimizer.zero_grad()
        activation_stats.clear()
        logits = model(input_ids)
        # logits: (B, T, V), targets: (B, T)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()

        global_step += 1

        if global_step % log_interval == 0:
            print(f"step={global_step}, epoch={epoch}, loss={loss.item():.4f}")
            log_param_stats(global_step, epoch)

print("Training loop completed. CSV stats at:", CSV_PATH)
print("If you want more/different stats, edit 'selected' list inside log_param_stats().")

# ----------------- End of script -----------------
