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
import umap
from sklearn.manifold import TSNE
from collections import defaultdict
import math
# Add imports for the new function
from pybbbc import BBBC021, constants
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import evaluate_model

def load_model_features(model_name, data_root="/scratch/cv-course2025/group8"):
    """
    Load features for a single model.
    
    Args:
        model_name (str): Name of the model (e.g., 'base_resnet', 'simclr', 'wsdino')
        data_root (str): Root directory containing bbbc021_features
        
    Returns:
        tuple: (features, moas) where features is numpy array and moas is list
    """
    feature_dir = os.path.join(data_root, "bbbc021_features", model_name)
    
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Features directory not found: {feature_dir}")
    
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

            features.append(feat.numpy() if hasattr(feat, 'numpy') else feat)
            moas.append(moa)

    if not features:
        raise ValueError(f"No valid features found for model {model_name}")
    
    return np.stack(features), moas

def plot_tsne_comparison(model_names, data_root="/scratch/cv-course2025/group8", output_dir="/scratch/cv-course2025/group8/plots", figsize=None):
    """
    Create side-by-side t-SNE plots for multiple models.
    
    Args:
        model_names (list): List of model names to compare
        data_root (str): Root directory containing bbbc021_features
        output_dir (str): Directory to save plots
        figsize (tuple): Figure size (width, height)
    """
    n_models = len(model_names)
    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Get all unique MoAs across all models for consistent coloring
    all_moas = set()
    model_data = {}
    
    for model_name in model_names:
        features, moas = load_model_features(model_name, data_root)
        model_data[model_name] = (features, moas)
        all_moas.update(moas)
    
    # Create consistent color mapping
    unique_moas = sorted(all_moas)
    moa_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_moas)))
    moa_to_color = {moa: moa_colors[i] for i, moa in enumerate(unique_moas)}
    
    for i, model_name in enumerate(model_names):
        features, moas = model_data[model_name]
        
        print(f"Running t-SNE for {model_name}...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_proj = tsne.fit_transform(features)
        
        ax = axes[i]
        
        # Plot each MoA separately for proper legend
        for moa in unique_moas:
            mask = np.array([m == moa for m in moas])
            if np.any(mask):
                ax.scatter(
                    tsne_proj[mask, 0], tsne_proj[mask, 1],
                    c=[moa_to_color[moa]], label=moa, alpha=0.7, s=30
                )
        
        ax.set_title(f"t-SNE: {model_name}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend only to the last subplot
        if i == n_models - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Hide any unused subplots
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"tsne_comparison_{'_'.join(model_names)}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"t-SNE comparison saved to {output_path}")

def plot_umap_comparison(model_names, data_root="/scratch/cv-course2025/group8", output_dir="/scratch/cv-course2025/group8/plots", figsize=None):
    """
    Create side-by-side UMAP plots for multiple models.
    
    Args:
        model_names (list): List of model names to compare
        data_root (str): Root directory containing bbbc021_features
        output_dir (str): Directory to save plots
        figsize (tuple): Figure size (width, height)
    """
    n_models = len(model_names)
    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Get all unique MoAs across all models for consistent coloring
    all_moas = set()
    model_data = {}
    
    for model_name in model_names:
        features, moas = load_model_features(model_name, data_root)
        model_data[model_name] = (features, moas)
        all_moas.update(moas)
    
    # Create consistent color mapping
    unique_moas = sorted(all_moas)
    moa_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_moas)))
    moa_to_color = {moa: moa_colors[i] for i, moa in enumerate(unique_moas)}
    
    for i, model_name in enumerate(model_names):
        features, moas = model_data[model_name]
        
        print(f"Running UMAP for {model_name}...")
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_proj = umap_reducer.fit_transform(features)
        
        ax = axes[i]
        
        # Plot each MoA separately for proper legend
        for moa in unique_moas:
            mask = np.array([m == moa for m in moas])
            if np.any(mask):
                ax.scatter(
                    umap_proj[mask, 0], umap_proj[mask, 1],
                    c=[moa_to_color[moa]], label=moa, alpha=0.7, s=30
                )
        
        ax.set_title(f"UMAP: {model_name}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend only to the last subplot
        if i == n_models - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Hide any unused subplots
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"umap_comparison_{'_'.join(model_names)}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"UMAP comparison saved to {output_path}")


def plot_accuracy_vs_image_count(model_names, data_root="/scratch/cv-course2025/group8", output_dir="/scratch/cv-course2025/group8/plots", distance_measure="cosine", sort_by="accuracy", max_image_count=None):
    """
    Create a dual-axis plot showing per-compound accuracies with image counts context.
    
    Args:
        model_names (list): List of model names to compare
        data_root (str): Root directory containing bbbc021_features and dataset
        output_dir (str): Directory to save plots
        distance_measure (str): Distance measure for evaluation ("cosine", "l1", "l2")
        sort_by (str): Sort compounds by "accuracy", "image_count", or "compound_name"
        max_image_count (int, optional): Maximum value for the image count y-axis (right side). If None, uses automatic scaling.
    """
    
    # First, collect image counts per compound
    print("Collecting image counts per compound...")
    compound_image_counts = defaultdict(int)
    
    # Get all compounds (excluding DMSO)
    compounds = [c for c in constants.COMPOUNDS if c != "DMSO"]
    moas = [m for m in constants.MOA if m not in ["DMSO", "null"]]
    
    # Count images per compound
    for compound in compounds:
        try:
            pybbbc = BBBC021(root_path=data_root, compound=compound, moa=moas)
            for image, meta in pybbbc:
                compound_name = meta.compound.compound
                if meta.compound.moa != 'null':  # Only count valid MoA samples
                    compound_image_counts[compound_name] += 1
        except Exception as e:
            print(f"Error processing compound {compound}: {e}")
            continue
    
    print(f"Found image counts for {len(compound_image_counts)} compounds")
    
    # Collect all model results first to determine consistent ordering
    all_model_results = {}
    all_compounds = set()
    
    for model_name in model_names:
        print(f"Evaluating model: {model_name}")
        try:
            results = evaluate_model(model_name, distance_measure=distance_measure, nsc_eval=True, tvn=False)
            
            # Extract compound accuracies
            compound_accuracies = {}
            for key, value in results.items():
                if key.startswith("compound_"):
                    compound = key.replace("compound_", "")
                    if compound in compound_image_counts:  # Only include compounds with image counts
                        compound_accuracies[compound] = value
                        all_compounds.add(compound)
            
            all_model_results[model_name] = compound_accuracies
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue
    
    # Determine consistent ordering based on the first model's results
    if not all_compounds:
        print("No compounds found across all models")
        return
    
    # Create a reference ordering based on sort_by parameter using the first model
    first_model = model_names[0]
    if first_model in all_model_results:
        reference_compounds = list(all_compounds)
        reference_accuracies = [all_model_results[first_model].get(comp, 0) for comp in reference_compounds]
        reference_image_counts = [compound_image_counts[comp] for comp in reference_compounds]
        
        # Sort based on the reference model
        if sort_by == "accuracy":
            sorted_indices = sorted(range(len(reference_compounds)), key=lambda i: reference_accuracies[i])
        elif sort_by == "image_count":
            sorted_indices = sorted(range(len(reference_compounds)), key=lambda i: reference_image_counts[i])
        else:  # sort by compound name
            sorted_indices = sorted(range(len(reference_compounds)), key=lambda i: reference_compounds[i])
        
        # Apply the same ordering to get consistent compound order
        compound_order = [reference_compounds[i] for i in sorted_indices]
    else:
        compound_order = sorted(all_compounds)  # Fallback to alphabetical
    
    # Create plots for each model
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 1, figsize=(15, 6*n_models))
    if n_models == 1:
        axes = [axes]
    
    for i, model_name in enumerate(model_names):
        if model_name not in all_model_results:
            continue
            
        compound_accuracies = all_model_results[model_name]
        
        # Use the consistent compound order
        compounds_sorted = compound_order
        accuracies_sorted = [compound_accuracies.get(comp, 0) for comp in compounds_sorted]
        image_counts_sorted = [compound_image_counts[comp] for comp in compounds_sorted]
        
        # Create dual-axis plot
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        # Plot accuracy bars
        x_pos = np.arange(len(compounds_sorted))
        bars1 = ax1.bar(x_pos, accuracies_sorted, alpha=0.7, color='steelblue', 
                       label='Accuracy', width=0.6)
        
        # Plot image count line
        line2 = ax2.plot(x_pos, image_counts_sorted, color='red', marker='o', 
                       linewidth=2, markersize=4, label='Image Count')
        
        # Formatting
        ax1.set_xlabel('Compounds', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12, color='steelblue')
        ax2.set_ylabel('Number of Images', fontsize=12, color='red')
        ax1.set_title(f'{model_name}: Per-Compound Accuracy vs Image Count ({distance_measure} distance)', 
                     fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(compounds_sorted, rotation=45, ha='right', fontsize=10)
        
        # Color the axes
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Set y-axis limits for image count if specified
        if max_image_count is not None:
            ax2.set_ylim(0, max_image_count)
        
        # Add grid
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars for readability
        for bar, acc in zip(bars1, accuracies_sorted):
            height = bar.get_height()
            ax1.annotate(f'{acc:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                       fontsize=8, rotation=90)
        
        # Add correlation coefficient
        correlation = np.corrcoef(accuracies_sorted, image_counts_sorted)[0, 1]
        ax1.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        print(f"Model {model_name}: Correlation between accuracy and image count: {correlation:.3f}")
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"accuracy_vs_image_count_{'_'.join(model_names)}_{distance_measure}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Dual-axis plot saved to {output_path}")
    
    # Print summary statistics using the consistent compound order
    all_image_counts = [compound_image_counts[comp] for comp in compound_order]
    print("\n=== Summary Statistics ===")
    print(f"Total compounds analyzed: {len(compound_order)}")
    print(f"Image count range: {min(all_image_counts)} - {max(all_image_counts)}")
    print(f"Mean images per compound: {np.mean(all_image_counts):.1f}")
    print(f"Compounds with <50 images: {sum(1 for c in all_image_counts if c < 50)}")
    print(f"Compounds with >200 images: {sum(1 for c in all_image_counts if c > 200)}")

if __name__ == "__main__":
    # Example: Compare multiple models
    #model_names = ["base_resnet", "simclr_vanilla_ws"]
    model_names = ["base_resnet", "wsdino"]
    
    output_dir=os.path.join("/scratch/cv-course2025/group8/plots", "wsdino")
    os.makedirs(output_dir, exist_ok=True)
    plot_accuracy_vs_image_count(model_names, sort_by="image_count", max_image_count=200, output_dir=output_dir)
    plot_umap_comparison(model_names, output_dir=output_dir)
    plot_tsne_comparison(model_names, output_dir=output_dir)