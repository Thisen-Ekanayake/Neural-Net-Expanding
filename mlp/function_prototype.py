import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Data
x = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
y = np.array([-0.523707628250122,
              -1.64705181121826,
              -2.18969106674194,
              -2.40047025680542,
              -2.52904319763184,
              -2.62181401252747,
              -2.69449019432068,
              -2.75430536270142,
              -2.8051929473877,
              -2.84952712059021])

# Exponential decay function
def exp_decay(x, A, k, C):
    return A * np.exp(-k * x) + C

# Fit the curve
params, _ = curve_fit(exp_decay, x, y, p0=(-0.5, 0.01, -2.9))
A, k, C = params
print(f"Fitted function: y = {A:.5f} * exp(-{k:.5f} * x) + {C:.5f}")

# Plot
plt.figure(figsize=(8,5))
plt.scatter(x, y, color='blue', label='Actual values', s=50)
plt.plot(x, exp_decay(x, *params), color='red', label='Fitted exponential', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('fc1.weight_0')
plt.title('Exponential Fit of fc1.weight_0')
plt.legend()
plt.grid(True)

# Save the graph
plt.savefig('fc1_weight0_fit.png', dpi=300)
plt.show()
