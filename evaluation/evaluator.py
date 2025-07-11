import torch
import pickle
import os
import numpy as np
from typing import Literal, Dict
import torch.nn.functional as F
from pybbbc import BBBC021, constants

from models.load_model import ModelName
from experiments.tvn import TypicalVariationNormalizer

DistanceMeasure = Literal["l1", "l2", "cosine"]

def evaluate_model(model_name: ModelName, distance_measure: DistanceMeasure = "cosine", nsc_eval = True, tvn: bool = False) -> Dict[str, float]:
    """
    Evaluate MOA prediction using 1-nearest neighbor with specified distance measure on pre-extracted features.
    
    Args:
        model_name: Name of the model to use for loading pre-computed features.
        distance_measure: Distance measure to use for 1NN ("l1", "l2", or "cosine").
        nsc_eval: If True, same compound (all concentrations) is not used for evaluation.
        tvn: If True, apply Typical Variation Normalization to features.
    
    Returns:
        Dict[str, float]: Dictionary with per-compound accuracies and total accuracy
    """
    
    # Load pre-computed features
    features_dir = f"/scratch/cv-course2025/group8/bbbc021_features/{model_name}"
    
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Load all feature files
    stored_features_dict = {}
    stored_keys = []
    
    for filename in os.listdir(features_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(features_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    key, features = pickle.load(f)
                    stored_features_dict[key] = features
                    stored_keys.append(key)  # (compound, concentration, moa)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    if not stored_features_dict:
        raise ValueError("No valid feature files found")
    
    # Apply TVN if requested
    if tvn:
        # Load DMSO features for fitting TVN from DMSO subfolder
        dmso_dir = os.path.join(features_dir, "DMSO")
        if not os.path.exists(dmso_dir):
            raise ValueError(f"DMSO directory not found: {dmso_dir}")
        
        dmso_features = []
        for filename in os.listdir(dmso_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(dmso_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        key, features = pickle.load(f)
                        dmso_features.append(features)
                except Exception as e:
                    print(f"Error loading DMSO features from {filepath}: {e}")
                    continue
        
        if not dmso_features:
            raise ValueError("No DMSO features found for TVN fitting")
        
        dmso_features = torch.stack(dmso_features)
        
        # Fit and transform features using TVN
        tvn_normalizer = TypicalVariationNormalizer()
        tvn_normalizer.fit(dmso_features)
        for key in stored_features_dict:
            stored_features_dict[key] = tvn_normalizer.transform(stored_features_dict[key].unsqueeze(0)).squeeze(0)
    
    # Track results
    total_correct = 0
    total_predictions = 0
    compound_results = {}
    
    # Process each treatment from stored features
    for treatment_key in stored_features_dict.keys():
        compound_name, concentration, true_moa = treatment_key
        
        # Skip DMSO compounds
        if compound_name == "DMSO":
            continue
            
        print(f"Evaluating treatment: {compound_name}@{concentration} ({true_moa})")
        
        # Create filtered reference features (exclude current compound if NSC is enabled)
        if nsc_eval:
            reference_keys = [key for key in stored_keys if key[0] != compound_name]
            if not reference_keys:
                print(f"Warning: No reference features available for compound {compound_name} with NSC evaluation. Skipping...")
                continue
        else:
            reference_keys = stored_keys
        
        # Find pre-computed features for this treatment
        treatment_key = (compound_name, concentration, true_moa)
        if treatment_key not in stored_features_dict:
            print(f"Warning: No pre-computed features found for treatment {compound_name}@{concentration}. Skipping...")
            continue
        
        # Get the stored features for this treatment
        avg_treatment_features = stored_features_dict[treatment_key]
        
        # Compute distances/similarities with average features
        best_score = float('-inf') if distance_measure == "cosine" else float('inf')
        best_idx = -1
        
        for i, ref_key in enumerate(reference_keys):
            ref_features = stored_features_dict[ref_key]
            
            if distance_measure == "cosine":
                features_norm = F.normalize(avg_treatment_features, p=2, dim=0)
                ref_features_norm = F.normalize(ref_features, p=2, dim=0)
                score = torch.dot(ref_features_norm, features_norm).item()
                if score > best_score:
                    best_score = score
                    best_idx = i
            elif distance_measure == "l2":
                score = torch.norm(ref_features - avg_treatment_features, p=2).item()
                if score < best_score:
                    best_score = score
                    best_idx = i
            elif distance_measure == "l1":
                score = torch.norm(ref_features - avg_treatment_features, p=1).item()
                if score < best_score:
                    best_score = score
                    best_idx = i
            else:
                raise ValueError(f"Unknown distance measure: {distance_measure}")
        
        # Get predicted MOA
        predicted_moa = reference_keys[best_idx][2]
        
        # Check if prediction is correct
        if predicted_moa == true_moa:
            total_correct += 1
        
        total_predictions += 1
        
        # Track per-compound results
        if compound_name not in compound_results:
            compound_results[compound_name] = {"correct": 0, "total": 0}
        
        compound_results[compound_name]["total"] += 1
        if predicted_moa == true_moa:
            compound_results[compound_name]["correct"] += 1
        
        print(f"  True MOA: {true_moa}, Predicted MOA: {predicted_moa}, Correct: {predicted_moa == true_moa}")
    
    # Calculate total accuracy
    total_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    
    # Calculate per-compound accuracies
    compound_accuracies = {}
    for compound, stats in compound_results.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        compound_accuracies[compound] = accuracy
    
    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"Model: {model_name}")
    print(f"Distance measure: {distance_measure}")
    print(f"NSC evaluation: {nsc_eval}")
    print(f"Total accuracy: {total_accuracy:.4f} ({total_correct}/{total_predictions})")
    print(f"\nPer-compound accuracies:")
    for compound, accuracy in sorted(compound_accuracies.items()):
        print(f"  {compound}: {accuracy:.4f} ({compound_results[compound]['correct']}/{compound_results[compound]['total']})")
    
    # Prepare return dictionary
    results = {
        "total_accuracy": total_accuracy,
        **{f"compound_{compound}": accuracy for compound, accuracy in compound_accuracies.items()}
    }
    
    return results

if __name__ == "__main__":
    model_name = "simclr"
    
    # Evaluate model
    results = evaluate_model(model_name, distance_measure="cosine", nsc_eval=True, tvn=False)
    
    print("\nEvaluation completed. Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")