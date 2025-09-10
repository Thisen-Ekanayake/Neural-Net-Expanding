import torch
import argparse
from collections import OrderedDict

def count_parameters(pt_file):
    checkpoint = torch.load(pt_file, map_location='cpu')
    
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        raise ValueError("No 'model_state' key found in checkpoint.")
    
    # Initialize counters
    total_params = 0
    arch_params = 0
    emb_params = 0
    head_params = 0
    layernorm_params = 0
    attention_params = 0
    ff_params = 0
    
    print("\nPer-parameter (raw listing):")
    for name, t in state_dict.items():
        num = t.numel()
        total_params += num

        # Categorize
        lname = name.lower()
        if 'emb' in lname:
            emb_params += num
        elif 'head' in lname:
            head_params += num
        elif 'ln' in lname:
            layernorm_params += num
            arch_params += num
        elif 'attn' in lname:
            attention_params += num
            arch_params += num
        elif 'ff' in lname or 'mlp' in lname:
            ff_params += num
            arch_params += num
        else:
            arch_params += num  # default to architecture

        print(f"{name:50s} {str(list(t.shape)):20s} -> {num:10d}")
    
    print("\nLayer-wise totals:")
    print(f"Positional Embedding: {emb_params if 'pos_emb' in state_dict else 0}")
    print(f"Embedding           : {emb_params}")
    print(f"LayerNorm           : {layernorm_params}")
    print(f"Attention           : {attention_params}")
    print(f"FeedForward         : {ff_params}")
    print(f"Output Head         : {head_params}")
    
    print(f"\nTotal parameters (including embeddings & head) : {total_params}")
    print(f"Architecture-only parameters (transformer layers only): {arch_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str, required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()
    
    count_parameters(args.pt)
