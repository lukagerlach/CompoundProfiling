"""
Visualize BBBC021 compound embeddings using t-SNE and UMAP.

This script loads pre-extracted feature embeddings from .pkl files,
applies t-SNE and UMAP dimensionality reduction, and visualizes the
resulting 2D projections colored by Mechanism of Action (MoA).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from collections import defaultdict

# --- Configuration
DATA_ROOT = "/scratch/cv-course2025/group8"
FEATURE_DIR = os.path.join(DATA_ROOT, "bbbc021_features", "base_resnet")

# --- Load Features from .pkl files
features = []
moas = []

for file in os.listdir(FEATURE_DIR):
    if not file.endswith(".pkl"):
        continue

    filepath = os.path.join(FEATURE_DIR, file)

    # Load (compound_name, concentration, moa), feature_vector
    with open(filepath, 'rb') as f:
        (compound_info, feat) = pickle.load(f)
        compound, conc, moa = compound_info

        # Skip samples with unknown MoA
        if moa == "null":
            continue

        features.append(feat.numpy())
        moas.append(moa)

# Convert to NumPy arrays
features = np.stack(features)
moas = np.array(moas)

# Create MoA → color index mapping
unique_moas = sorted(set(moas))
moa_to_color = {moa: i for i, moa in enumerate(unique_moas)}
colors = [moa_to_color[m] for m in moas]

# --- t-SNE Visualization
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_proj = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne_proj[:, 0], tsne_proj[:, 1],
    c=colors, cmap='tab20', alpha=0.7
)
plt.title("t-SNE of BBBC021 Features by MoA")
plt.colorbar(
    scatter,
    ticks=range(len(unique_moas)),
    label="MoA"
)
plt.clim(-0.5, len(unique_moas) - 0.5)
plt.xticks([])
plt.yticks([])
plt.savefig("tsne_moa.png")
plt.show()

# --- UMAP Visualization
print("Running UMAP...")
umap_proj = umap.UMAP(n_components=2, random_state=42).fit_transform(features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    umap_proj[:, 0], umap_proj[:, 1],
    c=colors, cmap='tab20', alpha=0.7
)
plt.title("UMAP of BBBC021 Features by MoA")
plt.colorbar(
    scatter,
    ticks=range(len(unique_moas)),
    label="MoA"
)
plt.clim(-0.5, len(unique_moas) - 0.5)
plt.xticks([])
plt.yticks([])
plt.savefig("umap_moa.png")
plt.show()
