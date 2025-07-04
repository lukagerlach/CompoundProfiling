from torchvision import models
import torch.nn as nn
import torch

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



def create_feature_extractor(pretrained_model: nn.Module) -> nn.Module:
    """Create a feature extractor from a pretrained ResNet50 model.
    Args:
        pretrained_model: Pretrained ResNet50 model
    Returns:
        nn.Module: Feature extractor model without the final classification layer
    """
    
    # Create feature extractor (remove final classification layer)
    # We basically cut off the last layer to get the feature extractor
    feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor