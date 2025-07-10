import torch
import pickle
import os
import numpy as np
from typing import Literal, Dict
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pybbbc import BBBC021, constants

from models.resnet_50_base import load_pretrained_model, create_feature_extractor, ModelName
from experiments.tvn import TypicalVariationNormalizer

DistanceMeasure = Literal["l1", "l2", "cosine"]

def evaluate_model(model_name: ModelName, data: BBBC021, device, distance_measure: DistanceMeasure = "cosine", nsc_eval = True, batch_size = 16, apply_tvn=False) -> Dict[str, float]:
    """
    Evaluate MOA prediction using 1-nearest neighbor with specified distance measure on pre-extracted features.
    
    Args:
        model_name: Name of the model to use for feature extraction.
        data: BBBC021 dataset object containing images and metadata.
        device: Device to run inference on (CPU or GPU).
        distance_measure: Distance measure to use for 1NN ("l1", "l2", or "cosine").
        nsc_eval: If True, same compound (all concentrations) is not used for evaluation.
        batch_size: Batch size for processing images.
    
    Returns:
        Dict[str, float]: Dictionary with per-compound accuracies and total accuracy
    """
    model = load_pretrained_model(model_name)
    feature_extractor = create_feature_extractor(model)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load pre-computed features
    features_dir = f"/scratch/cv-course2025/group8/bbbc021_features/{model_name}"
    
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Load all feature files
    stored_features = []
    stored_keys = []
    
    for filename in os.listdir(features_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(features_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    key, features = pickle.load(f)
                    stored_features.append(features)
                    stored_keys.append(key)  # (compound, concentration, moa)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    if not stored_features:
        raise ValueError("No valid feature files found")
    
    # Convert to tensor
    stored_features = torch.stack(stored_features)

    # TVN
    if apply_tvn:
        dmso_indices = [i for i, k in enumerate(stored_keys) if k[0] == "DMSO"]
        if not dmso_indices:
            raise ValueError("No DMSO (negative control) entries found for TVN.")
        dmso_features = torch.stack([stored_features[i] for i in dmso_indices])
        tvn = TypicalVariationNormalizer()
        tvn.fit(dmso_features)
        stored_features = torch.stack([tvn.transform(f) for f in stored_features])
    
    # Group evaluation data by treatment (compound × concentration)
    treatment_groups = {}
    for i, (image, metadata) in enumerate(data):
        if metadata.compound.moa == 'null':
            continue
        
        # Key: (compound, concentration, moa)
        key = (metadata.compound.compound, metadata.compound.concentration, metadata.compound.moa)
        if key not in treatment_groups:
            treatment_groups[key] = []
        treatment_groups[key].append((i, image, metadata))
    
    # Track results
    total_correct = 0
    total_predictions = 0
    compound_results = {}
    
    # Process each treatment
    for key, samples in treatment_groups.items():
        compound_name, concentration, true_moa = key
        print(f"Evaluating treatment: {compound_name}@{concentration} ({true_moa}) - {len(samples)} samples")
        
        # Create filtered reference features (exclude current compound if NSC is enabled)
        if nsc_eval:
            valid_indices = [i for i, ref_key in enumerate(stored_keys) if ref_key[0] != compound_name]
            if not valid_indices:
                print(f"Warning: No reference features available for compound {compound_name} with NSC evaluation. Skipping...")
                continue
            reference_features = stored_features[valid_indices]
            reference_keys = [stored_keys[i] for i in valid_indices]
        else:
            reference_features = stored_features
            reference_keys = stored_keys
        
        # Prepare images for this treatment
        images = []
        for _, image, metadata in samples:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            image = transform(image)
            images.append(image)
        
        # Create DataLoader for this treatment
        dataset = torch.utils.data.TensorDataset(torch.stack(images))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Extract features for all images of this treatment
        treatment_features = []
        with torch.no_grad():
            for batch in dataloader:
                batch_images = batch[0].to(device)
                features = feature_extractor(batch_images)
                features = features.squeeze()  # Remove spatial dimensions
                if len(features.shape) == 1:  # Handle single image case
                    features = features.unsqueeze(0)
                treatment_features.append(features.cpu())
        
        treatment_features = torch.cat(treatment_features, dim=0)
        
        # Compute average features for this treatment (like in extractor)
        avg_treatment_features = torch.mean(treatment_features, dim=0)
        
        # TVN
        if apply_tvn:
            avg_treatment_features = tvn.transform(avg_treatment_features)

        # Compute distances/similarities with average features
        if distance_measure == "cosine":
            features_norm = F.normalize(avg_treatment_features, p=2, dim=0)
            reference_features_norm = F.normalize(reference_features, p=2, dim=1)
            similarities = torch.mm(reference_features_norm, features_norm.unsqueeze(1)).squeeze()
            nearest_idx = torch.argmax(similarities).item()
            
        elif distance_measure == "l2":
            distances = torch.norm(reference_features - avg_treatment_features.unsqueeze(0), p=2, dim=1)
            nearest_idx = torch.argmin(distances).item()
            
        elif distance_measure == "l1":
            distances = torch.norm(reference_features - avg_treatment_features.unsqueeze(0), p=1, dim=1)
            nearest_idx = torch.argmin(distances).item()
            
        else:
            raise ValueError(f"Unknown distance measure: {distance_measure}")
        
        # Get predicted MOA
        predicted_moa = reference_keys[nearest_idx][2]
        
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
    # Example usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    model_name = "base_resnet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BBBC021 dataset
    data_root = "/scratch/cv-course2025/group8"
    compounds = constants.COMPOUNDS
    compounds.remove("DMSO")
    data = BBBC021(root_path=data_root, compound=compounds)  # Load all compounds
    
    # Evaluate model
    results = evaluate_model(model_name, data, device, distance_measure="cosine", nsc_eval=True, batch_size=64)
    
    print("\nEvaluation completed. Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")