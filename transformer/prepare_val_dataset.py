#!/usr/bin/env python3
"""
Prepare validation dataset:
1. Clean and preprocess raw text.
2. Train SentencePiece (if requested).
3. Tokenize into JSONL format (train/val split).
"""

import os
import sys
import argparse
import random
import json
import subprocess
from tqdm import tqdm

# ===== CONFIG (import your training config if needed) =====
CONFIG = {
    "block_size": 128,   # tokens per block
}

# ===== Preprocessing: split long lines =====
def preprocess_text(in_path, out_path, max_chars=4000):
    """Split overly long lines into chunks (SentencePiece limit)."""
    with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="preprocessing"):
            line = line.strip()
            if not line:
                continue
            while len(line) > max_chars:
                fout.write(line[:max_chars] + "\n")
                line = line[max_chars:]
            if line:
                fout.write(line + "\n")
    print(f"Preprocessing complete. Cleaned file: {out_path}")

# ===== Train SentencePiece =====
def train_sentencepiece(raw_txt_path, model_prefix,
                        vocab_size=4000, model_type='unigram',
                        character_coverage=1.0,
                        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
                        input_sentence_size=1_000_000,
                        seed_sentencepiece_size=1_000_000,
                        max_sentence_length=200000):
    """Train SentencePiece safely using all CPU cores, with progress logging."""
    num_threads = os.cpu_count() or 8

    # special tokens must be unique
    if len({unk_id, bos_id, eos_id, pad_id}) < 4:
        raise RuntimeError("Special token IDs must be distinct!")

    cmd = [
        "spm_train",
        f"--input={raw_txt_path}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        f"--unk_id={unk_id}",
        f"--bos_id={bos_id}",
        f"--eos_id={eos_id}",
        f"--pad_id={pad_id}",
        "--shuffle_input_sentence=1",
        f"--input_sentence_size={input_sentence_size}",
        f"--seed_sentencepiece_size={seed_sentencepiece_size}",
        f"--num_threads={num_threads}",
        f"--max_sentence_length={max_sentence_length}",
    ]

    print(f"Training SentencePiece with {num_threads} threads...")
    print(" ".join(cmd))

    # run as subprocess to capture logs & show progress
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    pbar = tqdm(desc="SentencePiece training", unit=" steps", dynamic_ncols=True)
    for line in process.stdout:
        pbar.update(1)  # bump progress bar per log line
        if "loss" in line.lower() or "epoch" in line.lower():
            pbar.set_postfix_str(line.strip()[:80])
    process.wait()
    pbar.close()

    if process.returncode != 0:
        raise RuntimeError("SentencePiece training failed")

    model_path = model_prefix + ".model"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")
    return model_path

# ===== Tokenize dataset =====
def tokenize_dataset(sp_model_path, txt_path, out_jsonl, block_size, val_fraction=0.01):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    lines = open(txt_path, "r", encoding="utf-8").read().splitlines()
    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - val_fraction))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    def write_blocks(lines, out_path):
        with open(out_path, "w", encoding="utf-8") as fout:
            buf = []
            for line in tqdm(lines, desc=f"tokenizing {out_path}"):
                ids = sp.encode(line, out_type=int)
                for t in ids:
                    buf.append(t)
                    if len(buf) >= block_size:
                        fout.write(json.dumps({"input_ids": buf}) + "\n")
                        buf = []
            if buf:
                fout.write(json.dumps({"input_ids": buf}) + "\n")

    write_blocks(train_lines, "train_" + out_jsonl)
    write_blocks(val_lines, "val_" + out_jsonl)
    print(f"Done. Train/val written to train_{out_jsonl}, val_{out_jsonl}")

# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_txt", required=True, help="Path to raw .txt file")
    parser.add_argument("--train_sp", action="store_true", help="Train SentencePiece model")
    parser.add_argument("--sp_model_out", default="sp", help="SentencePiece model prefix")
    parser.add_argument("--out_jsonl", default="dataset.jsonl", help="Output JSONL filename")
    parser.add_argument("--val_fraction", type=float, default=0.01, help="Validation fraction")
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--model_type", default="unigram")
    parser.add_argument("--max_chars", type=int, default=4000, help="Max chars per line after split")
    parser.add_argument("--max_sentence_length", type=int, default=200000, help="SP trainer max sentence length")
    args = parser.parse_args()

    cleaned = args.raw_txt + ".cleaned"
    preprocess_text(args.raw_txt, cleaned, max_chars=args.max_chars)

    sp_model_path = args.sp_model_out + ".model"
    if args.train_sp or not os.path.exists(sp_model_path):
        sp_model_path = train_sentencepiece(cleaned, args.sp_model_out,
                                            vocab_size=args.vocab_size,
                                            model_type=args.model_type,
                                            max_sentence_length=args.max_sentence_length)

    tokenize_dataset(sp_model_path, cleaned, args.out_jsonl,
                     block_size=CONFIG["block_size"],
                     val_fraction=args.val_fraction)

if __name__ == "__main__":
    main()
