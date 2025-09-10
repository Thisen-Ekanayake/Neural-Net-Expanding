import torch

"""pt_file = "checkpoint/final.pt"  # change to your path
checkpoint = torch.load(pt_file, map_location="cpu")

print("Type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Keys:", checkpoint.keys())
    for k, v in checkpoint.items():
        print(f"{k}: {type(v)}")"""

import torch

def count_parameters(pt_file):
    checkpoint = torch.load(pt_file, map_location="cpu")

    if "model_state" not in checkpoint:
        raise ValueError("No 'model_state' found in checkpoint")

    state_dict = checkpoint["model_state"]

    total_params = 0
    for name, param in state_dict.items():
        tensor = torch.tensor(param) if not torch.is_tensor(param) else param
        num_params = tensor.numel()
        print(f"{name:50s} {list(tensor.shape)} -> {num_params}")
        total_params += num_params

    print("\nTotal parameters:", total_params)

if __name__ == "__main__":
    pt_file = "checkpoint/final.pt"  # 🔹 replace with your file path
    count_parameters(pt_file)
