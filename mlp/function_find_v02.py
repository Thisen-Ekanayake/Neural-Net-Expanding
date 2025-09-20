import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example data
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2, 4, 6, 8, 10, 12, 14])

# Define candidate functions
def linear(x, a, b): 
    return a*x + b

def quadratic(x, a, b, c): 
    return a*x**2 + b*x + c

def cubic(x, a, b, c, d): 
    return a*x**3 + b*x**2 + c*x + d

def exponential(x, a, b): 
    return a * np.exp(b*x)

def logarithmic(x, a, b): 
    return a * np.log(x) + b

# Candidate models
models = {
    "Linear": linear,
    "Quadratic": quadratic,
    "Cubic": cubic,
    "Exponential": exponential,
    "Logarithmic": logarithmic
}

best_model = None
best_score = float("inf")
fit_results = {}

# Try each model
for name, func in models.items():
    try:
        popt, _ = curve_fit(func, x, y, maxfev=5000)  # fit
        y_pred = func(x, *popt)
        error = np.mean((y - y_pred)**2)  # mean squared error
        fit_results[name] = (popt, error)
        
        if error < best_score:
            best_score = error
            best_model = (name, func, popt)
    except:
        continue  # skip models that fail to fit

# Print results
print("Best model:", best_model[0])
print("Parameters:", best_model[2])

# Plot best fit
plt.scatter(x, y, color="red", label="Data points")
x_fit = np.linspace(min(x), max(x), 200)
y_fit = best_model[1](x_fit, *best_model[2])
plt.plot(x_fit, y_fit, label=f"Best fit: {best_model[0]}")
plt.legend()
plt.show()
