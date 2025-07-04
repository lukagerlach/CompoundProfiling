import pickle
import torch
from torch.utils.data import DataLoader
import numpy as np

from models.resnet_50_base import load_pretrained_resnet50, create_feature_extractor
from pybbbc import BBBC021, constants

import os
from collections import defaultdict

def extract_moa_features(device, batch_size = 16, data_root: str = "/scratch/cv-course2025/group8", moas: list[str] = []) -> None:
    """
    Extract features for the BBBC021 dataset using a pretrained ResNet50 model.
    
    Args:
        data_root: Root directory where the BBBC021 dataset is stored.
        moas: List of Mechanisms of Action (MOAs) to filter the dataset.
        If None, all MOAs will be processed. If ['null'], only images with
        no MOA will be processed.
        
    Note:
        This function assumes that the BBBC021 dataset has already been downloaded
        and preprocessed.
    """
    
    # Load pretrained ResNet50 model
    pretrained_model = load_pretrained_resnet50()
    # Create feature extractor
    feature_extractor = create_feature_extractor(pretrained_model)
    
    if not moas:
        # If no MOAs specified or empty list, process all available MOAs
        moas = constants.MOA
    else:
        # check if moas is subset of constants.MOA
        for moa in moas:
            if moa not in constants.MOA:
                raise ValueError(f"MOA '{moa}' is not a valid Mechanism of Action. "
                                 f"Valid MOAs are: {constants.MOA}")
    
    # Create output directory
    output_dir = os.path.join(data_root, "bbbc021_features", "baseline")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = device
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Process each MOA
    for moa in moas:
        moa_images = BBBC021(root_path=data_root, moa=moa).images
        print(f"Processing MOA: {moa} with {len(moa_images)} images")
        
        dataloader = DataLoader(moa_images, batch_size=batch_size, shuffle=False)
        # TODO: check if we want features per trwatment of per moa
        # per treatment would be more fine-grained and allows nsc to be used i guess?
        # this way we could also just aggregate treatments into moas
        
        
# main function to run the feature extraction
if __name__ == "__main__":
    extract_moa_features(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                          batch_size=16, 
                          data_root="/scratch/cv-course2025/group8", 
                          moas=['Actin disruptors'])