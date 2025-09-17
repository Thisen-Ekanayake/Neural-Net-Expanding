import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import traceback

# ==== Load CSV ====
df = pd.read_csv("mlp_params_raw_50_epoch.csv")  # adjust path/delimiter if needed
x = df["epoch"].values.astype(float)

# optional: scale x to improve numeric stability for exponential fits
# comment out if you want to use raw epoch values
# x = (x - x.min()) / (x.max() - x.min() + 1e-12)

# ==== Candidate models ====
def linear(x, a, b):
    return a * x + b

def polynomial2(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b):
    # a * exp(b * x) — be careful if x is large
    return a * np.exp(b * x)

# Make logarithmic model include an offset c (learned, constrained > 0)
def logarithmic_with_offset(x, a, b, c):
    # we require x + c > 0; bounds will enforce c > 0
    return a * np.log(x + c) + b

# ==== Models config (function, initial guess, bounds) ====
models = {
    "linear": {
        "func": linear,
        "p0": [0.0, 0.0],
        "bounds": (-np.inf, np.inf),
    },
    "quadratic": {
        "func": polynomial2,
        "p0": [0.0, 0.0, 0.0],
        "bounds": (-np.inf, np.inf),
    },
    "exponential": {
        "func": exponential,
        "p0": [1.0, 0.01],
        # small bounds on b to avoid crazy explosions (optional)
        "bounds": ([-np.inf, -np.inf], [np.inf, np.inf]),
    },
    # The crucial change: add offset c and force c > 0
    "logarithmic": {
        "func": logarithmic_with_offset,
        "p0": [1.0, 0.0, 1.0],  # a, b, c
        "bounds": ([-np.inf, -np.inf, 1e-8], [np.inf, np.inf, np.inf]),
    },
}

# ==== Output directory for plots ====
outdir = "fit_results_50_epoch"
os.makedirs(outdir, exist_ok=True)

# ==== Iterate over each weight column ====
results_summary = {}

for col in df.columns[1:]:  # skip 'epoch' column
    y = df[col].values.astype(float)
    col_results = {}

    # quick check: if all y are nan or constant, skip or handle specially
    if np.all(np.isnan(y)):
        print(f"{col}: all NaN, skipping.")
        continue

    for name, spec in models.items():
        func = spec["func"]
        p0 = spec.get("p0", None)
        bounds = spec.get("bounds", (-np.inf, np.inf))

        try:
            # call curve_fit with bounds if provided as tuple/list
            if bounds is None:
                popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=20000)
            else:
                popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)

            y_pred = func(x, *popt)

            # if prediction contains invalid values, treat as failed fit
            if not np.all(np.isfinite(y_pred)):
                raise ValueError("Prediction contains non-finite values.")

            mse = mean_squared_error(y, y_pred)
            # R^2 for extra diagnostic
            denom = np.sum((y - np.nanmean(y)) ** 2)
            r2 = 1.0 - np.sum((y - y_pred) ** 2) / denom if denom > 0 else np.nan

            col_results[name] = {
                "success": True,
                "mse": float(mse),
                "r2": float(r2) if np.isfinite(r2) else None,
                "popt": popt,
                "pcov": pcov,
                "error": None,
            }

        except Exception as e:
            # save error message for debugging
            col_results[name] = {
                "success": False,
                "mse": np.inf,
                "r2": None,
                "popt": None,
                "pcov": None,
                "error": traceback.format_exc(limit=1),
            }

    # choose best model by minimal MSE (only among successful fits)
    best_model = None
    best_mse = np.inf
    for name, info in col_results.items():
        if info["success"] and info["mse"] < best_mse and np.isfinite(info["mse"]):
            best_mse = info["mse"]
            best_model = name

    # fallback: if none succeeded, pick the one with smallest mse value (may be inf)
    if best_model is None:
        best_model = min(col_results, key=lambda k: col_results[k]["mse"])

    results_summary[col] = (best_model, col_results[best_model])

    # ==== Plot ====
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, label="data", zorder=3)
    x_sorted_idx = np.argsort(x)
    x_plot = x[x_sorted_idx]

    for name, info in col_results.items():
        if info["success"] and np.isfinite(info["mse"]):
            y_plot = models[name]["func"](x_plot, *info["popt"])
            plt.plot(x_plot, y_plot, label=f"{name} (MSE={info['mse']:.4g}, R²={info['r2']:.3f})", zorder=2)

    plt.title(f"{col} best fit: {best_model}")
    plt.xlabel("epoch")
    plt.ylabel(col)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{col}.png"))
    plt.close()

# ==== Print summary ====
for col, (best_model, info) in results_summary.items():
    print(f"{col}: Best fit = {best_model}")
    if info["success"]:
        print(f"    MSE = {info['mse']:.6g}, R² = {info['r2']}")
        print(f"    Params = {info['popt']}")
    else:
        print("    No successful fit. Last error for that model:")
        print(info["error"])
