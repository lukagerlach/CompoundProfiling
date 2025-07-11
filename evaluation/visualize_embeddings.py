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

def visualize_embeddings(feature_dir, output_dir=".", model_name="base_resnet"):
    """
    Visualize compound embeddings using t-SNE and UMAP.
    
    Args:
        feature_dir (str): Directory containing .pkl feature files
        output_dir (str): Directory to save visualization plots
        model_name (str): Name of the model for plot titles
    """
    # --- Load Features from .pkl files
    features = []
    moas = []

    for file in os.listdir(feature_dir):
        if not file.endswith(".pkl") or "DMSO" in file:
            continue

        filepath = os.path.join(feature_dir, file)

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

    print(f"Loaded {len(features)} features with {len(unique_moas)} unique MoAs")
    print(f"MoAs: {unique_moas}")

    # --- t-SNE Visualization
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_proj = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_proj[:, 0], tsne_proj[:, 1],
        c=colors, cmap='tab20', alpha=0.7
    )
    plt.title(f"t-SNE of {model_name} Features by MoA")
    plt.colorbar(
        scatter,
        ticks=range(len(unique_moas)),
        label="MoA"
    )
    plt.clim(-0.5, len(unique_moas) - 0.5)
    plt.xticks([])
    plt.yticks([])
    tsne_path = os.path.join(output_dir, f"tsne_{model_name}.png")
    plt.savefig(tsne_path)
    plt.show()
    print(f"t-SNE plot saved to {tsne_path}")

    # --- UMAP Visualization
    print("Running UMAP...")
    umap_proj = umap.UMAP(n_components=2, random_state=42).fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        umap_proj[:, 0], umap_proj[:, 1],
        c=colors, cmap='tab20', alpha=0.7
    )
    plt.title(f"UMAP of {model_name} Features by MoA")
    plt.colorbar(
        scatter,
        ticks=range(len(unique_moas)),
        label="MoA"
    )
    plt.clim(-0.5, len(unique_moas) - 0.5)
    plt.xticks([])
    plt.yticks([])
    umap_path = os.path.join(output_dir, f"umap_{model_name}.png")
    plt.savefig(umap_path)
    plt.show()
    print(f"UMAP plot saved to {umap_path}")

if __name__ == "__main__":
    # --- Configuration
    DATA_ROOT = "/scratch/cv-course2025/group8"
    FEATURE_DIR = os.path.join(DATA_ROOT, "bbbc021_features", "simclr")
    OUTPUT_DIR = "."
    MODEL_NAME = "simclr"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run visualization
    visualize_embeddings(FEATURE_DIR, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)