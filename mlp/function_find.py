import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# ==== Load CSV ====
df = pd.read_csv("mlp_params_raw.csv")  # adjust delimiter if needed
x = df["epoch"].values

# ==== Candidate models ====
def linear(x, a, b):
    return a*x + b

def polynomial2(x, a, b, c):
    return a*x**2 + b*x + c

def exponential(x, a, b):
    return a * np.exp(b*x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

models = {
    "linear": (linear, [1, 1]),
    "quadratic": (polynomial2, [1, 1, 1]),
    "exponential": (exponential, [1, 0.01]),
    "logarithmic": (logarithmic, [1, 1]),
}

# ==== Output directory for plots ====
os.makedirs("fit_results", exist_ok=True)

# ==== Iterate over each weight column ====
results_summary = {}

for col in df.columns[1:]:  # skip 'epoch'
    y = df[col].values
    col_results = {}

    for name, (func, guess) in models.items():
        try:
            popt, _ = curve_fit(func, x, y, p0=guess, maxfev=10000)
            y_pred = func(x, *popt)
            mse = mean_squared_error(y, y_pred)
            col_results[name] = (mse, popt)
        except:
            col_results[name] = (np.inf, None)

    # Find best model
    best_model = min(col_results, key=lambda k: col_results[k][0])
    results_summary[col] = (best_model, col_results[best_model])

    # ==== Plot ====
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="data", color="black")
    for name, (mse, popt) in col_results.items():
        if popt is not None and np.isfinite(mse):
            plt.plot(x, models[name][0](x, *popt), label=f"{name} (MSE={mse:.3f})")
    plt.title(f"{col} best fit: {best_model}")
    plt.xlabel("epoch")
    plt.ylabel(col)
    plt.legend()
    plt.savefig(f"fit_results/{col}.png")
    plt.close()

# ==== Print summary ====
for col, (best_model, (mse, popt)) in results_summary.items():
    print(f"{col}: Best fit = {best_model}, Params = {popt}, MSE={mse:.4f}")
