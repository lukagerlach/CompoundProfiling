from typing import Literal
from torchvision import models
import torch.nn as nn
import torch

ModelName = Literal["base_resnet", "simclr", "wsdino"]

def load_pretrained_model(model_name: ModelName, weight_path='/scratch/cv-course2025/group8/model_weights'):
    """Load pretrained ResNet50 model."""
    
    # Load full model
    if model_name == "base_resnet":
        return load_pretrained_resnet50(weights="IMAGENET1K_V2")
        
    elif model_name == "simclr":
        return load_pretrained_model_from_weights("resnet50_simclr", weight_path)
    
    elif model_name == "wsdino":
        return load_pretrained_model_from_weights("resnet50_wsdino", weight_path)

def load_pretrained_resnet50(weights: str = "IMAGENET1K_V2") -> object:
    """Load pretrained ResNet50 model.
    
    Args:
        weights: Model weights to use
        
    Note:
        See https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
        for available weights.
    """
    print("Loading pretrained ResNet50...")
    
    # Load full model
    if weights == "IMAGENET1K_V2":
        pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        try:
            pretrained_model = models.resnet50(weights=weights)
        except Exception as e:
            raise ValueError(f"Failed to load ResNet50 with weights '{weights}': {e}")

    pretrained_model.eval()
    return pretrained_model

def load_pretrained_model_from_weights(model_name: str, weight_path: str) -> nn.Module:
    """Load pretrained ResNet50 model from custom weights.
    
    Args:
        model_name: Name of the model to load
        weight_path: Path to the weights file
        
    Returns:
        nn.Module: Pretrained ResNet50 model
    """
    print(f"Loading pretrained ResNet50 from {weight_path}...")
    
    # Load the model architecture
    pretrained_model = models.resnet50(weights=None)
    
    # Load the weights
    try:
        state_dict = torch.load(f"{weight_path}/{model_name}.pth", map_location='cpu')
        
        # Check if this is a backbone/feature extractor state dict (missing fc layer)
        model_keys = set(pretrained_model.state_dict().keys())
        saved_keys = set(state_dict.keys())
        missing_fc = any('fc.' in key for key in (model_keys - saved_keys))
        
        if missing_fc:
            print("Loading backbone weights (without final classification layer)...")
            # Load backbone weights, keep randomly initialized fc layer
            pretrained_model.load_state_dict(state_dict, strict=False)
        else:
            # Standard loading for complete model
            pretrained_model.load_state_dict(state_dict)
            
    except FileNotFoundError:
        raise ValueError(f"Weight file '{model_name}.pth' not found in '{weight_path}'")
    except Exception as e:
        raise ValueError(f"Error loading weights: {e}")
    
    pretrained_model.eval()
    return pretrained_model

def create_feature_extractor(pretrained_model: nn.Module) -> nn.Module:
    """Create a feature extractor from a pretrained ResNet50 model.
    Args:
        pretrained_model: Pretrained ResNet50 model
    Returns:
        nn.Module: Feature extractor model without the final classification layer
    """
    
    # Create feature extractor (remove final classification layer)
    feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor