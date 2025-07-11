from enum import Enum
import pickle
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os

from models.load_model import load_pretrained_model, create_feature_extractor, ModelName
from pybbbc import BBBC021, constants
from experiments.tvn import TypicalVariationNormalizer  # Import TVN module

def extract_moa_features(
        model_name: ModelName, 
        device, 
        batch_size=16, 
        data_root: str = "/scratch/cv-course2025/group8", 
        compounds: list[str] = None, 
        tvn: bool = False
) -> None:
    """
    Extract features for the BBBC021 dataset using a pretrained ResNet50 model.
    
    Args:
        model_name: Name of the model to use. Is of type ModelName.
        device: Device to run the model on.
        batch_size: Batch size for data loading.
        data_root: Root directory where the BBBC021 dataset is stored.
        compounds: List of compounds to process. If None, all compounds will be processed.
        tvn: If True, apply Typical Variation Normalization to features before averaging and saving.
    """
    
    # Load pretrained ResNet50 model
    pretrained_model = load_pretrained_model(model_name)
    # Create feature extractor
    feature_extractor = create_feature_extractor(pretrained_model)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    if not compounds:
        compounds = constants.COMPOUNDS
    else:
        for compound in compounds:
            if compound not in constants.COMPOUNDS:
                raise ValueError(f"Compound '{compound}' is not valid. Valid compounds: {constants.COMPOUNDS}")

    # Output directory
    output_dir = os.path.join(data_root, "bbbc021_features", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DMSO subfolder
    dmso_dir = os.path.join(output_dir, "DMSO")
    os.makedirs(dmso_dir, exist_ok=True)
    
    # Set device
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Collect per-compound features
    compound_features = {}
    tvn_features = []

    # Process each compound dynamically
    for compound in compounds:
        data = BBBC021(root_path=data_root, compound=compound)
        print(f"Processing Compound: {compound} with {len(data.images)} images")

        # Dictionary to store images grouped by (compound, concentration, moa)
        image_groups: Dict[Tuple[str, float, str], list[torch.Tensor]] = {}
        
        # Collect images for this compound
        for image, metadata in data:
            if metadata.compound.moa == 'null':
                print(f"Skipping image with null MOA for compound {compound}.")
                continue
            
            key = (metadata.compound.compound, 
                   metadata.compound.concentration,
                   metadata.compound.moa)
            
            if key not in image_groups:
                image_groups[key] = []

            # Convert numpy array to tensor if needed
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            image = transform(image)
            image_groups[key].append(image)
        
        # Process each group for this compound immediately
        for key, images in image_groups.items():
            compound_name, concentration, moa = key
            
            if len(images) == 0:
                print(f"Warning: No images for group {compound_name}_{concentration}. Skipping...")
                continue
                
            print(f"Extracting features for {compound_name}@{concentration}({moa}) - {len(images)} images")
            
            try:
                # Create DataLoader for this group
                dataset = torch.utils.data.TensorDataset(torch.stack(images))
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                # Extract features
                all_features = []
                with torch.no_grad():
                    for batch in dataloader:
                        batch_images = batch[0].to(device)
                        features = feature_extractor(batch_images)
                        features = features.squeeze()  # Remove spatial dimensions
                        if len(features.shape) == 1:
                            features = features.unsqueeze(0)
                        all_features.append(features.cpu())
                
                all_features = torch.cat(all_features, dim=0)

                if tvn and compound_name == "DMSO":
                    tvn_features.append(all_features)  # Store for TVN fitting

                compound_features[key] = all_features  # Store per-image features for every treatment

            except Exception as e:
                print(f"Error processing group {compound_name}_{concentration}: {e}. Skipping...")
                continue

    # TVN logic
    if tvn:
        print("\nFitting TVN from DMSO images...")
        if not tvn_features:
            raise RuntimeError("No DMSO features found to fit TVN.")
        dmso_concat = torch.cat(tvn_features, dim=0)
        tvn = TypicalVariationNormalizer()
        tvn.fit(dmso_concat)

        print("Applying TVN and saving averaged features...")
        for key, features in compound_features.items():
            transformed = tvn.transform(features)
            avg_feature = torch.mean(transformed, dim=0)
            result = (key, avg_feature)
            compound_name, concentration, _ = key
            filename = f"{compound_name}_{concentration}.pkl".replace(" ", "_").replace("/", "_")
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved averaged TVN features to {filepath}")

    else:
        print("Saving non-TVN averaged features...")
        for key, features in compound_features.items():
            avg_feature = torch.mean(features, dim=0)
            result = (key, avg_feature)
            compound_name, concentration, _ = key
            filename = f"{compound_name}_{concentration}.pkl".replace(" ", "_").replace("/", "_")
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved averaged features to {filepath}")

# main function to run the feature extraction
if __name__ == "__main__":
    extract_moa_features(
        model_name="simclr", 
        # model_name="wsdino",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=128,
        data_root="/scratch/cv-course2025/group8",
        compounds=constants.COMPOUNDS,
        tvn=True
    )