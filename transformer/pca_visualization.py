# this is an example for pca visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# 1. Simulate fake "weights"
# -------------------------
# Imagine a layer with 50 weights
# We track them for 100 "training steps"
steps = 100
weights = np.zeros((steps, 50))

# Create a trajectory: two strong directions of change
t = np.linspace(0, 4*np.pi, steps)
weights[:, 0] = np.sin(t) * 5          # direction 1
weights[:, 1] = np.cos(t) * 3          # direction 2
weights[:, 2] = np.linspace(0, 10, steps)  # slow drift
# Add noise
weights += np.random.randn(*weights.shape) * 0.5

print("Original shape:", weights.shape)  # (100, 50)

# -------------------------
# 2. Apply PCA
# -------------------------
pca = PCA(n_components=2)  # reduce 50D → 2D
latent = pca.fit_transform(weights)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# -------------------------
# 3. Plot comparisons
# -------------------------

# Plot raw weights (just the first 3 dims for visualization)
plt.figure(figsize=(10,4))
plt.plot(weights[:,0], label="Weight dim 1")
plt.plot(weights[:,1], label="Weight dim 2")
plt.plot(weights[:,2], label="Weight dim 3")
plt.title("Original high-dimensional weight updates (sample dims)")
plt.xlabel("Training step")
plt.ylabel("Weight value")
plt.legend()
plt.show()

# PCA latent trajectory
plt.figure(figsize=(6,6))
plt.plot(latent[:,0], latent[:,1], marker='o')
plt.title("Weight evolution compressed with PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# PCA time series
plt.figure(figsize=(10,4))
plt.plot(latent[:,0], label="PC1")
plt.plot(latent[:,1], label="PC2")
plt.title("PCA coordinates over time")
plt.xlabel("Training step")
plt.ylabel("PCA coeff")
plt.legend()
plt.show()
