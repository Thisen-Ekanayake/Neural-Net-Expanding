# -------------------------------
# Analyze PCA layer_splines.npz
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load("pca/layer_splines.npz", allow_pickle=True)

print("Layers stored:", len(data.files))

# -------------------------------
# 1. Plot trajectories for one layer
# -------------------------------
def plot_layer(key, dims=3):
    cs, pca = data[key]
    latents = np.array([cs(step) for step in cs.x])
    
    plt.figure(figsize=(10, 5))
    for i in range(min(dims, latents.shape[1])):
        plt.plot(cs.x, latents[:, i], label=f"PC{i+1}")
    
    plt.title(f"Latent Trajectory of {key}")
    plt.xlabel("Training step")
    plt.ylabel("PCA coefficient value")
    plt.legend()
    plt.show()

# Example: first layer
plot_layer(list(data.files)[0], dims=3)

# -------------------------------
# 2. Compare variance explained
# -------------------------------
def plot_variance(key):
    _, pca = data[key]
    var = pca.explained_variance_ratio_
    plt.figure(figsize=(6,4))
    plt.bar(range(1, len(var)+1), var*100)
    plt.title(f"PCA Variance Explained: {key}")
    plt.xlabel("PC dimension")
    plt.ylabel("% variance")
    plt.show()

plot_variance(list(data.files)[0])

# -------------------------------
# 3. Heatmap: which layers changed most
# -------------------------------
layer_changes = {}
for key in data.files:
    cs, _ = data[key]
    latents = np.array([cs(step) for step in cs.x])
    total_change = np.linalg.norm(latents[-1] - latents[0])  # L2 distance in latent space
    layer_changes[key] = total_change

# Sort layers by change magnitude
sorted_layers = sorted(layer_changes.items(), key=lambda x: -x[1])

plt.figure(figsize=(10, 6))
plt.barh([k for k, _ in sorted_layers[:20]], [v for _, v in sorted_layers[:20]])
plt.title("Top 20 Layers with Most Change (latent space L2)")
plt.xlabel("Change magnitude")
plt.gca().invert_yaxis()
plt.show()

# -------------------------------
# 4. Optional: interactive browsing
# -------------------------------
def browse_layers(n=5):
    for key in list(data.files)[:n]:
        print("\n---", key, "---")
        plot_layer(key, dims=3)
        plot_variance(key)

# browse_layers(3)  # uncomment to view 3 layers
