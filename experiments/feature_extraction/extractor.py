from enum import Enum
import pickle
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

from models.resnet_50_base import load_pretrained_model, create_feature_extractor, ModelName
from pybbbc import BBBC021, constants

import os


def extract_moa_features(model_name: ModelName, device, batch_size = 16, data_root: str = "/scratch/cv-course2025/group8", compounds: list[str] = None) -> None:
    """
    Extract features for the BBBC021 dataset using a pretrained ResNet50 model.
    
    Args:
        model_name: Name of the model to use. Is of type MODEL_NAMES.
        device: Device to run the model on
        batch_size: Batch size for data loading
        data_root: Root directory where the BBBC021 dataset is stored.
        compounds: List of compounds to process. If None, all compounds will be processed.
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
                raise ValueError(f"Compound '{compound}' is not a valid compound. "
                                 f"Valid compounds are: {constants.COMPOUNDS}")
    
    # Create output directory with model name
    output_dir = os.path.join(data_root, "bbbc021_features", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Process each compound dynamically
    for compound in compounds:        
        data = BBBC021(root_path=data_root, compound=compound)  # Fixed: use single compound
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
            # Apply transforms (resize + normalize) - THIS IS MISSING!
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
                        all_features.append(features.cpu())
                
                # Compute average features
                all_features = torch.cat(all_features, dim=0)
                avg_features = torch.mean(all_features, dim=0)
                
                # Create result as tuple (key, feature)
                result = (key, avg_features)
                
                # Create filename: compound_concentration
                filename = f"{compound_name}_{concentration}.pkl".replace(' ', '_').replace('/', '_')
                filepath = os.path.join(output_dir, filename)
                
                # Save to file
                with open(filepath, 'wb') as f:
                    pickle.dump(result, f)
                
                print(f"Saved features to {filepath}")
                
            except Exception as e:
                print(f"Error processing group {compound_name}_{concentration}: {e}. Skipping...")
                continue


# main function to run the feature extraction
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    extract_moa_features(
        model_name="base_resnet",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        batch_size=32, 
        data_root="/scratch/cv-course2025/group8",
        compounds=constants.COMPOUNDS)