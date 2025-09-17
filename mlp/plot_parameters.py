import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
csv_file = "mlp_params_raw.csv"  # replace with your CSV file path
output_folder = "plots"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

# Plot each column (except 'epoch') against 'epoch'
for col in df.columns:
    if col == "epoch":
        continue

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df[col], marker='o', linestyle='-', markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel(col)
    plt.title(f"{col} vs Epoch")
    plt.grid(True)

    # Safe filename (replace spaces or special chars)
    filename = f"{col.replace(' ', '_').replace('.', '_')}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.close()

print(f"Plots saved in folder '{output_folder}'")
