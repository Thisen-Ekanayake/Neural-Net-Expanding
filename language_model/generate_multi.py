import torch
from transformers import GPT2LMHeadModel
import sentencepiece as spm

PROMPT_FILE = "prompts.txt"
OUTPUT_FILE = "generated-multi.txt"
TOKENIZER_MODEL = ""
MODEL_PATH = "final-model"

# --- Load Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Tokenizer ---
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_MODEL)

# --- Load Trained Model ---
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

# --- Load Prompts ---
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# --- Generate & Save Output ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
    for i, prompt in enumerate(prompts, 1):
        input_ids = torch.tensor([sp.EncodeAsIds(prompt)]).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=sp.PieceToId("<pad>") if sp.PieceToId("<pad>") != -1 else 0,
                eos_token_id=sp.PieceToId("</s>") if sp.PieceToId("</s>") != -1 else 2,
                no_repeat_ngram_size=2,
            )

        generated_text = sp.DecodeIds(output_ids[0].tolist())

        print(f"\n[{i}] Prompt: {prompt}")
        print(f"Generated: {generated_text}")

        out_file.write(f"=== Prompt {i} ===\n{prompt}\n\n")
        out_file.write(f"=== Output {i} ===\n{generated_text}\n\n{'='*60}\n\n")

print(f"\nAll outputs saved to {OUTPUT_FILE}")