import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.7, 7.4, 20.1, 54.6, 148.4])  # looks exponential

# Define candidate functions
def linear(x, a, b):
    return a*x + b

def polynomial2(x, a, b, c):  # quadratic
    return a*x**2 + b*x + c

def exponential(x, a, b):
    return a * np.exp(b*x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

# Fit each model
models = {
    "linear": (linear, [1, 1]),
    "quadratic": (polynomial2, [1, 1, 1]),
    "exponential": (exponential, [1, 0.1]),
    "logarithmic": (logarithmic, [1, 1])
}

results = {}
for name, (func, guess) in models.items():
    try:
        popt, _ = curve_fit(func, x, y, p0=guess, maxfev=10000)
        y_pred = func(x, *popt)
        mse = mean_squared_error(y, y_pred)
        results[name] = (mse, popt)
    except:
        results[name] = (np.inf, None)  # fitting failed

# Find best model
best_model = min(results, key=lambda k: results[k][0])

print("Best fit model:", best_model)
print("Parameters:", results[best_model][1])

# Plot
plt.scatter(x, y, label="data")
for name, (mse, popt) in results.items():
    if popt is not None:
        plt.plot(x, models[name][0](x, *popt), label=f"{name} (MSE={mse:.2f})")
plt.legend()
plt.show()
