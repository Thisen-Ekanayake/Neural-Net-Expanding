import sentencepiece as spm
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------------------------
# CONFIGURATION
# ---------------------------
DATASET_PATH = "100M_words.txt"  # Replace with your dataset path
SP_MODEL_PATH = "unigram_4000_0.9995.model"  # Replace with your trained SentencePiece model
OUTPUT_JSONL = "tokenized_dataset.jsonl"
BLOCK_SIZE = 512  # Token block length
CHUNK_LINES = 10000  # Number of lines to read at once to save memory

# ---------------------------
# INITIALIZE TOKENIZER
# ---------------------------
sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)

# ---------------------------
# TOKENIZATION FUNCTION
# ---------------------------
def tokenize_chunk(lines):
    """
    Tokenizes a list of lines and splits into blocks of BLOCK_SIZE.
    Returns a list of dictionaries: {"input_ids": [token_ids...]}
    """
    tokenized_blocks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        token_ids = sp.encode(line, out_type=int)
        # Split long token sequences into BLOCK_SIZE chunks
        for i in range(0, len(token_ids), BLOCK_SIZE):
            block = token_ids[i:i + BLOCK_SIZE]
            tokenized_blocks.append({"input_ids": block})
    return tokenized_blocks

# ---------------------------
# MULTIPROCESSING HELPER
# ---------------------------
def process_chunk(chunk_lines):
    return tokenize_chunk(chunk_lines)

# ---------------------------
# MAIN TOKENIZATION LOOP
# ---------------------------
def tokenize_dataset():
    dataset_path = Path(DATASET_PATH)
    output_path = Path(OUTPUT_JSONL)

    with dataset_path.open("r", encoding="utf-8") as f, output_path.open("w", encoding="utf-8") as out_file:
        pool = Pool(processes=cpu_count())
        chunk_lines = []

        for line in tqdm(f, desc="Reading lines"):
            chunk_lines.append(line)
            if len(chunk_lines) >= CHUNK_LINES:
                # Tokenize in parallel
                results = pool.map(process_chunk, [chunk_lines])
                for token_blocks in results:
                    for block in token_blocks:
                        out_file.write(json.dumps(block) + "\n")
                chunk_lines = []

        # Process remaining lines
        if chunk_lines:
            results = pool.map(process_chunk, [chunk_lines])
            for token_blocks in results:
                for block in token_blocks:
                    out_file.write(json.dumps(block) + "\n")

        pool.close()
        pool.join()

if __name__ == "__main__":
    tokenize_dataset()
