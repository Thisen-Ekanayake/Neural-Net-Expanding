"""
analyze_npz.py

Script to analyze and visualize parameter evolution
from .npz model checkpoints dumped during training.

- Loads all .npz files from nn_cache/
- Extracts weight statistics (mean, std, min, max, L2 norm)
- Plots these stats across steps for selected parameters
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

CACHE_DIR = "nn_cache"  # folder where .npz files are stored
OUT_DIR = "analysis_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. Collect .npz files and sort by step ---
files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npz")]
def extract_step(fname):
    m = re.search(r"step_(\d+)", fname)
    return int(m.group(1)) if m else -1
files = sorted(files, key=extract_step)

print(f"Found {len(files)} checkpoint files")

# --- 2. Select a few parameters to track ---
# You can adjust this list based on model structure
PARAMS_TO_TRACK = [
    "tok_emb_weight",
    "transformer_layer_0_linear1_weight",
    "transformer_layer_0_linear2_weight",
    "transformer_layer_0_self_attn_in_proj_weight",
    "head_weight"
]

# --- 3. Extract stats per checkpoint ---
stats = {p: {"steps": [], "mean": [], "std": [], "l2": [], "min": [], "max": []} for p in PARAMS_TO_TRACK}

for fname in files:
    step = extract_step(fname)
    path = os.path.join(CACHE_DIR, fname)
    data = np.load(path)

    for param in PARAMS_TO_TRACK:
        if param not in data:  # skip if not present
            continue
        arr = data[param].astype(np.float32)
        stats[param]["steps"].append(step)
        stats[param]["mean"].append(arr.mean())
        stats[param]["std"].append(arr.std())
        stats[param]["min"].append(arr.min())
        stats[param]["max"].append(arr.max())
        stats[param]["l2"].append(np.linalg.norm(arr))

print("Finished collecting stats.")

# --- 4. Plot each stat ---
for param, sdict in stats.items():
    if not sdict["steps"]:
        continue
    steps = sdict["steps"]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(steps, sdict["mean"], marker="o")
    plt.title(f"{param} mean")

    plt.subplot(2, 2, 2)
    plt.plot(steps, sdict["std"], marker="o")
    plt.title(f"{param} std")

    plt.subplot(2, 2, 3)
    plt.plot(steps, sdict["l2"], marker="o")
    plt.title(f"{param} L2 norm")

    plt.subplot(2, 2, 4)
    plt.plot(steps, sdict["min"], marker="o", label="min")
    plt.plot(steps, sdict["max"], marker="o", label="max")
    plt.title(f"{param} min/max")
    plt.legend()

    plt.suptitle(f"Parameter evolution: {param}")
    out_path = os.path.join(OUT_DIR, f"{param}_stats.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot for {param} -> {out_path}")

print("All plots saved in:", OUT_DIR)
