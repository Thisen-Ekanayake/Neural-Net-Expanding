import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- CONFIG ---
csv_file = "mlp_params_raw.csv"  # replace with your CSV path
output_folder = "plots_exp_log_poly"
equation_file = os.path.join(output_folder, "fitted_equations.txt")
polynomial_degree = 3  # for polynomial fit

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

# Define function forms
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def logarithmic(x, a, b, c):
    return a * np.log(x + 1e-8) + b  # +1e-8 to avoid log(0)

# Open file to save equations
with open(equation_file, "w") as eq_file:

    for col in df.columns:
        if col == "epoch":
            continue

        x = df["epoch"].values
        y = df[col].values

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o', label='Original', markersize=3)

        # --- Polynomial Fit ---
        poly_coeffs = np.polyfit(x, y, polynomial_degree)
        poly = np.poly1d(poly_coeffs)
        y_poly = poly(x)
        plt.plot(x, y_poly, '-', label=f'Polynomial deg={polynomial_degree}')
        eq_file.write(f"{col} Polynomial: {poly}\n")

        # --- Exponential Fit ---
        try:
            exp_params, _ = curve_fit(exponential, x, y, maxfev=10000)
            y_exp = exponential(x, *exp_params)
            plt.plot(x, y_exp, '-', label='Exponential')
            eq_file.write(f"{col} Exponential: a={exp_params[0]:.4f}, b={exp_params[1]:.4f}, c={exp_params[2]:.4f}\n")
        except Exception as e:
            eq_file.write(f"{col} Exponential fit failed: {e}\n")

        # --- Logarithmic Fit ---
        try:
            log_params, _ = curve_fit(logarithmic, x, y, maxfev=10000)
            y_log = logarithmic(x, *log_params)
            plt.plot(x, y_log, '-', label='Logarithmic')
            eq_file.write(f"{col} Logarithmic: a={log_params[0]:.4f}, b={log_params[1]:.4f}\n")
        except Exception as e:
            eq_file.write(f"{col} Logarithmic fit failed: {e}\n")

        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(f"{col} vs Epoch")
        plt.legend()
        plt.grid(True)

        # Save plot
        safe_filename = f"{col.replace(' ', '_').replace('.', '_')}.png"
        plt.savefig(os.path.join(output_folder, safe_filename))
        plt.close()
        eq_file.write("\n")  # blank line between columns

print(f"All plots saved in '{output_folder}' and equations in '{equation_file}'")
