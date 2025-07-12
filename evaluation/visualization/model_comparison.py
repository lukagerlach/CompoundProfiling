"""
Model comparison visualizations for BBBC021 compound profiling.

This module provides functions to compare different model performances
including accuracy metrics, confusion matrices, and similarity analysis.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict

# Add parent directory to path to import evaluator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import evaluate_model

def load_model_features_and_moas(model_name, data_root="/scratch/cv-course2025/group8"):
    """
    Load features and MoAs for a single model.
    
    Args:
        model_name (str): Name of the model
        data_root (str): Root directory containing bbbc021_features
        
    Returns:
        tuple: (features, moas) numpy arrays
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

        try:
            with open(filepath, 'rb') as f:
                (compound_info, feat) = pickle.load(f)
                compound, conc, moa = compound_info

                # Skip samples with unknown MoA
                if moa == "null":
                    continue

                features.append(feat.numpy() if hasattr(feat, 'numpy') else feat)
                moas.append(moa)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    if not features:
        raise ValueError(f"No valid features found for model {model_name}")
    
    return np.stack(features), np.array(moas)

def plot_accuracy_comparison(model_names, data_root="/scratch/cv-course2025/group8", output_dir="/scratch/cv-course2025/group8/plots", distance_measure="cosine"):
    """
    Create a bar chart comparing total accuracy across models.
    
    Args:
        model_names (list): List of model names to compare
        data_root (str): Root directory containing features
        output_dir (str): Directory to save plots
        distance_measure (str): Distance measure for evaluation
    """
    accuracies = {}
    
    for model_name in model_names:
        try:
            print(f"Evaluating {model_name}...")
            results = evaluate_model(model_name, distance_measure=distance_measure, nsc_eval=True, tvn=False)
            accuracies[model_name] = results['total_accuracy']
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            accuracies[model_name] = 0.0
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    models = list(accuracies.keys())
    accuracy_values = [accuracies[model] for model in models]
    
    bars = plt.bar(models, accuracy_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracy_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{accuracy:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title(f'Model Accuracy Comparison ({distance_measure} distance)', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, max(accuracy_values) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"accuracy_comparison_{distance_measure}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Accuracy comparison saved to {output_path}")
    
    return accuracies

def plot_confusion_matrices(model_names, data_root="/scratch/cv-course2025/group8", output_dir="/scratch/cv-course2025/group8/plots", distance_measure="cosine"):
    """
    Create confusion matrices for each model showing predicted vs actual MoA.
    
    Args:
        model_names (list): List of model names to compare
        data_root (str): Root directory containing features
        output_dir (str): Directory to save plots
        distance_measure (str): Distance measure for evaluation
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, model_name in enumerate(model_names):
        try:
            print(f"Creating confusion matrix for {model_name}...")
            
            # Load features and MoAs
            features, moas = load_model_features_and_moas(model_name, data_root)
            
            # Get unique MoAs
            unique_moas = sorted(set(moas))
            
            # Perform 1-NN classification
            if distance_measure == "cosine":
                # Normalize features for cosine similarity
                features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
                nbrs = NearestNeighbors(n_neighbors=2, metric='cosine')
                nbrs.fit(features_norm)
                distances, indices = nbrs.kneighbors(features_norm)
            else:
                nbrs = NearestNeighbors(n_neighbors=2, metric=distance_measure)
                nbrs.fit(features)
                distances, indices = nbrs.kneighbors(features)
            
            # Get predictions (nearest neighbor that's not itself)
            predicted_moas = []
            actual_moas = []
            
            for j, (dist, idx) in enumerate(zip(distances, indices)):
                # Skip self (first neighbor)
                nearest_idx = idx[1]
                predicted_moas.append(moas[nearest_idx])
                actual_moas.append(moas[j])
            
            # Create confusion matrix
            cm = confusion_matrix(actual_moas, predicted_moas, labels=unique_moas)
            
            # Plot
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=unique_moas, yticklabels=unique_moas,
                       ax=ax, cbar=i == n_models - 1)  # Only show colorbar for last plot
            
            ax.set_title(f'{model_name}')
            ax.set_xlabel('Predicted MoA')
            ax.set_ylabel('Actual MoA')
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
        except Exception as e:
            print(f"Error creating confusion matrix for {model_name}: {e}")
            ax = axes[i]
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{model_name} - Error')
    
    plt.suptitle(f'Confusion Matrices ({distance_measure} distance)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"confusion_matrices_{distance_measure}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Confusion matrices saved to {output_path}")

def plot_similarity_comparison(model_names, data_root="/scratch/cv-course2025/group8", output_dir="/scratch/cv-course2025/group8/plots"):
    """
    Compare same-MoA vs different-MoA cosine similarities across models.
    
    Args:
        model_names (list): List of model names to compare
        data_root (str): Root directory containing features
        output_dir (str): Directory to save plots
    """
    # Create subplots: bar chart + distribution plots
    fig = plt.figure(figsize=(15, 10))
    
    # Top: Bar chart comparison
    ax_bar = plt.subplot(2, 2, (1, 2))
    
    # Bottom: Distribution plots
    n_models = len(model_names)
    
    same_moa_means = []
    diff_moa_means = []
    model_labels = []
    
    for i, model_name in enumerate(model_names):
        try:
            print(f"Analyzing similarities for {model_name}...")
            
            # Load features and MoAs
            features, moas = load_model_features_and_moas(model_name, data_root)
            
            # Normalize features for cosine similarity
            features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            # Compute pairwise cosine similarities
            similarity_matrix = cosine_similarity(features_norm)
            
            # Extract same-MoA and different-MoA similarities
            same_moa_sims = []
            diff_moa_sims = []
            
            for j in range(len(features)):
                for k in range(j + 1, len(features)):
                    sim = similarity_matrix[j, k]
                    if moas[j] == moas[k]:
                        same_moa_sims.append(sim)
                    else:
                        diff_moa_sims.append(sim)
            
            # Store means for bar chart
            same_moa_means.append(np.mean(same_moa_sims))
            diff_moa_means.append(np.mean(diff_moa_sims))
            model_labels.append(model_name)
            
            # Plot distribution for this model
            ax_dist = plt.subplot(2, n_models, n_models + i + 1)
            
            # Create histograms
            ax_dist.hist(same_moa_sims, bins=50, alpha=0.7, label='Same MoA', 
                        color='green')
            ax_dist.hist(diff_moa_sims, bins=50, alpha=0.7, label='Different MoA', 
                        color='red')
            
            ax_dist.set_title(f'{model_name}')
            ax_dist.set_xlabel('Cosine Similarity')
            ax_dist.set_ylabel('Frequency')
            ax_dist.legend()
            ax_dist.grid(alpha=0.3)
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            same_moa_means.append(0)
            diff_moa_means.append(0)
            model_labels.append(model_name)
    
    # Create bar chart
    x = np.arange(len(model_labels))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, same_moa_means, width, label='Same MoA', color='green', alpha=0.7)
    bars2 = ax_bar.bar(x + width/2, diff_moa_means, width, label='Different MoA', color='red', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax_bar.set_xlabel('Model')
    ax_bar.set_ylabel('Average Cosine Similarity')
    ax_bar.set_title('Same vs Different MoA Similarity Comparison')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(model_labels)
    ax_bar.legend()
    ax_bar.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "similarity_comparison.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Similarity comparison saved to {output_path}")
    
    # Return summary statistics
    return {
        'models': model_labels,
        'same_moa_means': same_moa_means,
        'diff_moa_means': diff_moa_means,
        'separations': [same - diff for same, diff in zip(same_moa_means, diff_moa_means)]
    }

def compare_all_models(model_names, data_root="/scratch/cv-course2025/group8", output_dir="/scratch/cv-course2025/group8/plots", distance_measure="cosine"):
    """
    Run all model comparison visualizations.
    
    Args:
        model_names (list): List of model names to compare
        data_root (str): Root directory containing features
        output_dir (str): Directory to save plots
        distance_measure (str): Distance measure for evaluation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Model Comparison Analysis ===")
    print(f"Models: {model_names}")
    print(f"Distance measure: {distance_measure}")
    print()
    
    # 1. Accuracy comparison
    print("1. Creating accuracy comparison...")
    accuracies = plot_accuracy_comparison(model_names, data_root, output_dir, distance_measure)
    print()
    
    # 2. Confusion matrices
    print("2. Creating confusion matrices...")
    plot_confusion_matrices(model_names, data_root, output_dir, distance_measure)
    print()
    
    # 3. Similarity analysis
    print("3. Creating similarity analysis...")
    similarity_stats = plot_similarity_comparison(model_names, data_root, output_dir)
    print()
    
    # Print summary
    print("=== Summary ===")
    print(f"Accuracies: {accuracies}")
    print(f"MoA separations: {dict(zip(similarity_stats['models'], similarity_stats['separations']))}")
    print()
    print("All visualizations completed!")

if __name__ == "__main__":
    # Models to compare
    model_names = ["base_resnet", "simclr_vanilla_ws"]
    
    # Run all comparisons
    compare_all_models(model_names)
