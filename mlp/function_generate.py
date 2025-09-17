import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid GTK crash
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
csv_file = "mlp_params_raw.csv"  # replace with your CSV file path
output_folder = "plots"
equation_file = os.path.join(output_folder, "fitted_equations.txt")
polynomial_degree = 3  # adjust as needed

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

# Open a file to save equations
with open(equation_file, "w") as eq_file:

    # Loop through all columns except 'epoch'
    for col in df.columns:
        if col == "epoch":
            continue

        x = df["epoch"].values
        y = df[col].values

        # Fit polynomial
        coeffs = np.polyfit(x, y, polynomial_degree)
        poly = np.poly1d(coeffs)
        y_fit = poly(x)

        # Plot original + fitted curve
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o', label='Original', markersize=3)
        plt.plot(x, y_fit, '-', label=f'Poly deg={polynomial_degree}')
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(f"{col} vs Epoch")
        plt.legend()
        plt.grid(True)

        # Save plot
        safe_filename = f"{col.replace(' ', '_').replace('.', '_')}.png"
        plt.savefig(os.path.join(output_folder, safe_filename))
        plt.close()

        # Save equation
        eq_file.write(f"{col} = {poly}\n\n")

print(f"All plots saved in '{output_folder}' and equations in '{equation_file}'")
