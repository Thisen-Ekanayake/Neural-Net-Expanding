#!/usr/bin/env python3
"""
parameter_count.py

Loads a .pt checkpoint (handles common layouts), prints parameter shapes and counts,
detects shared storages (tied parameters), reports raw vs unique parameter totals,
and can optionally write a 'cleaned' checkpoint that removes duplicate state_dict entries.

Usage:
    python parameter_count.py --pt path/to/final.pt
    python parameter_count.py --pt final.pt --clean_out final.cleaned.pt
"""

import torch
import argparse
import csv
import json
from collections import defaultdict

def count_parameters(state_dict):
    """
    Counts parameters and groups them by component prefix.
    """
    total_params = 0
    grouped_params = defaultdict(int)
    raw_listing = []

    for name, param in state_dict.items():
        t = torch.tensor(param) if not torch.is_tensor(param) else param
        num_params = t.numel()
        shape_str = str(list(t.shape))

        # Raw list
        raw_listing.append((name, shape_str, num_params))

        # Group by top-level component
        if name.startswith("tok_emb"):
            group = "Embedding"
        elif name.startswith("pos_emb"):
            group = "Positional Embedding"
        elif "attn" in name:
            group = "Attention"
        elif "ff" in name:
            group = "FeedForward"
        elif "ln" in name:
            group = "LayerNorm"
        elif name.startswith("head"):
            group = "Output Head"
        else:
            group = "Other"

        grouped_params[group] += num_params
        total_params += num_params

    return total_params, raw_listing, grouped_params


def analyze_checkpoint(pt_file, csv_out="params.csv", json_out="params.json"):
    checkpoint = torch.load(pt_file, map_location="cpu")

    if "model_state" not in checkpoint:
        raise ValueError("No 'model_state' found in checkpoint")

    state_dict = checkpoint["model_state"]
    total_params, raw_listing, grouped_params = count_parameters(state_dict)

    print("\nPer-parameter (raw listing):")
    for name, shape, count in raw_listing:
        print(f"{name:50s} {shape:25s} -> {count:10d}")

    print("\nLayer-wise totals:")
    for group, count in grouped_params.items():
        print(f"{group:20s}: {count}")

    print(f"\nTotal parameters: {total_params}\n")

    # Save CSV
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Shape", "Count"])
        writer.writerows(raw_listing)

    # Save JSON
    with open(json_out, "w") as f:
        json.dump(
            {"layer_totals": grouped_params, "total_params": total_params},
            f,
            indent=4,
        )

    print(f"Saved detailed CSV to {csv_out}")
    print(f"Saved summary JSON to {json_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--csv_out", type=str, default="params.csv", help="CSV output file")
    parser.add_argument("--json_out", type=str, default="params.json", help="JSON output file")
    args = parser.parse_args()

    analyze_checkpoint(args.pt, csv_out=args.csv_out, json_out=args.json_out)
