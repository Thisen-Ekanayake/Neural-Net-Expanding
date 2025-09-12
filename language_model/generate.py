import os
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import sentencepiece as spm

def generate_text(model_path, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model config and model
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")
    config = GPT2Config.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, config=config).to(device)
    model.eval()

    # Load SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(" ")

    # Encode prompt
    input_ids = torch.tensor([sp.EncodeAsIds(prompt)]).to(device)

    # Generate tokens
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=sp.PieceToId("<pad>") if sp.PieceToId("<pad>") != -1 else 0,
            eos_token_id=sp.PieceToId("</s>") if sp.PieceToId("</s>") != -1 else 2,
            no_repeat_ngram_size=2,
        )

    # Decode output
    generated = sp.DecodeIds(output_ids[0].tolist())

    print("\n=== Generated Text ===\n")
    print(generated)

    # Optional: Save to file
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(generated)
        print(f"\nSaved generated text to: {save_path}")

    return generated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sinhala text using your fine-tuned GPT model")
    parser.add_argument("--model_path", type=str, default="final-model", help="Path to trained model folder")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to start generation")
    parser.add_argument("--max_length", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--save_path", type=str, help="Optional file path to save generated output")

    args = parser.parse_args()

    generate_text(
        model_path=args.model_path,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        save_path=args.save_path
    )
