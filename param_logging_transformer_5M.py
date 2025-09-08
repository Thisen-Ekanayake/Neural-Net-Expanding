"""
param_logging_transformer_5M_pretokenized.py

Modified training pipeline:
 - Uses a pre-tokenized dataset in JSONL format with 512-token sequences.
 - Pads/truncates sequences to 512 tokens.
 - Removes all SentencePiece/tokenization logic.
 - Keeps the Transformer LM (~5M params) and detailed parameter/activation logging.
"""

import os
import time
import math
import csv
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- PLACEHOLDERS ---
DATASET_JSONL_PATH = "tokenized_dataset.jsonl"  # <-- your pre-tokenized JSONL
CACHE_DIR = "nn_cache"  # <-- caching / checkpoints / dumps
# --------------------

VOCAB_SIZE = 4000
BLOCK_SIZE = 512
BATCH_SIZE = 48
NUM_EPOCHS = 3
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_interval = 5
full_dump_interval = 500
NUM_WORKERS = os.cpu_count() or 1
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------- Pre-tokenized dataset loader (with padding) -----------------
class PreTokenizedDataset(Dataset):
    def __init__(self, jsonl_path, block_size=512, pad_token=0):
        self.blocks = []
        self.block_size = block_size
        self.pad_token = pad_token
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                ids = data['input_ids']
                # pad if shorter
                if len(ids) < block_size:
                    ids = ids + [pad_token] * (block_size - len(ids))
                elif len(ids) > block_size:
                    ids = ids[:block_size]  # truncate if longer
                self.blocks.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        x = self.blocks[idx]
        return x[:-1], x[1:]  # input and target

# ----------------- Transformer LM -----------------
class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layer=6, n_head=8, d_ff=None, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, d_model)
        d_ff = d_ff or (4 * d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = x.transpose(0, 1)  # Transformer expects (T, B, C)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ----------------- Parameter breakdown -----------------
def print_param_breakdown(model):
    print("Parameter count breakdown (module -> numel):")
    breakdown = []
    for name, module in model.named_modules():
        if len(list(module.parameters(recurse=False))) == 0:
            continue
        module_params = sum(p.numel() for p in module.parameters(recurse=False))
        if module_params > 0:
            breakdown.append((name or "<root>", module_params))
    total = sum(p.numel() for p in model.parameters())
    for name, cnt in breakdown:
        print(f"  {name:40s} : {cnt:,d}")
    print(f"Total parameters: {total:,d}")
    return total, breakdown

# ----------------- Build model -----------------
D_MODEL = 384
N_LAYER = 6
N_HEAD = 8
D_FF = 4 * D_MODEL

model = SimpleTransformerLM(VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, d_ff=D_FF).to(DEVICE)
total_params, breakdown = print_param_breakdown(model)
print(f"Model built with d_model={D_MODEL}, n_layer={N_LAYER}, n_head={N_HEAD}. Total params: {total_params:,d}")

# ----------------- Logging -----------------
CSV_PATH = os.path.join(CACHE_DIR, "param_stats.csv")
csv_fieldnames = [
    "timestamp", "step", "epoch",
    "scope", "layer_name", "param_name", "param_type", "shape",
    "stat_type", "value"
]
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
        writer.writeheader()

def tensor_stats(tensor):
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

activation_stats = {}
def make_forward_hook(name):
    def hook(module, input, output):
        try:
            out = output[0] if isinstance(output, tuple) else output
            activation_stats[name] = tensor_stats(out.view(-1))
        except Exception as e:
            activation_stats[name] = {"error": str(e)}
    return hook

for i, layer in enumerate(model.transformer.layers):
    layer_name = f"transformer.layer.{i}"
    if hasattr(layer, 'linear1'):
        layer.linear1.register_forward_hook(make_forward_hook(layer_name + ".ffn_in"))
    if hasattr(layer, 'linear2'):
        layer.linear2.register_forward_hook(make_forward_hook(layer_name + ".ffn_out"))

# ----------------- Dataset and DataLoader -----------------
print(f"Using {NUM_WORKERS} DataLoader workers.")
train_dataset = PreTokenizedDataset(DATASET_JSONL_PATH, block_size=BLOCK_SIZE, pad_token=0)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

# ----------------- Training -----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()
global_step = 0

def log_param_stats(step, epoch):
    rows = []
    ts = time.time()
    total_param_norm = 0.0
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        scope, _, param_name = name.rpartition('.')
        p_stats = tensor_stats(param.data)
        g_stats = tensor_stats(param.grad) if param.grad is not None else None
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

    rows.append({
        'timestamp': ts, 'step': step, 'epoch': epoch,
        'scope': '<global>', 'layer_name': '<global>', 'param_name': '<global>',
        'param_type': 'global', 'shape': '', 'stat_type': 'total_param_l2_norm',
        'value': total_param_norm
    })
    rows.append({
        'timestamp': ts, 'step': step, 'epoch': epoch,
        'scope': '<global>', 'layer_name': '<global>', 'param_name': '<global>',
        'param_type': 'global', 'shape': '', 'stat_type': 'total_grad_l2_norm',
        'value': total_grad_norm
    })

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

    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
        for r in rows:
            writer.writerow(r)

    if full_dump_interval and step % full_dump_interval == 0:
        dump_path = os.path.join(CACHE_DIR, f"raw_weights_step_{step}.npz")
        print(f"Dumping raw weights to {dump_path}")
        dump_dict = {name.replace('.', '_'): param.detach().cpu().numpy() for name, param in model.named_parameters()}
        import numpy as np
        np.savez_compressed(dump_path, **dump_dict)

# ----------------- Training loop -----------------
model.train()
for epoch in range(1, NUM_EPOCHS + 1):
    for batch in train_loader:
        input_ids, target_ids = batch
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)

        optimizer.zero_grad()
        activation_stats.clear()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()

        global_step += 1
        if global_step % log_interval == 0:
            print(f"step={global_step}, epoch={epoch}, loss={loss.item():.4f}")
            log_param_stats(global_step, epoch)

print("Training loop completed. CSV stats at:", CSV_PATH)
