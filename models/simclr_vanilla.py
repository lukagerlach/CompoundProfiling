import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

from pybbbc import BBBC021, constants

class SimCLRProjectionHead(nn.Module):
    """Projection head for SimCLR"""
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class SimCLRModel(nn.Module):
    """
        This is our model for vanilla SimCLR,
        with ResNet50 backbone.
    """
    def __init__(self, backbone_model, projection_dim=128):
        super().__init__()
        self.backbone = backbone_model
        
        # Remove the final classification layer
        if hasattr(self.backbone, 'fc'):
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_dim = 2048
            
        self.projection_head = SimCLRProjectionHead(
            input_dim=backbone_dim, 
            output_dim=projection_dim
        )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        # Project features
        projections = self.projection_head(features)
        return features, projections


class SimCLRVanillaDataset(Dataset):
    """
    Dataset for (vanilla) SimCLR: returns two augmentations of the same image.
    Optionally returns compound labels for compound-aware training.
    """
    def __init__(self, root_path, transform=None, compound_aware=False):
        self.root_path = root_path
        self.transform = transform
        self.compound_aware = compound_aware
        
        # Create a basic resize transform for memory efficiency
        # We'll handle tensor conversion manually since images might already be tensors
        self.resize_transform = transforms.Resize((224, 224))
        
        moas = constants.MOA.copy()
        if "null" in moas:
            moas.remove("null")
        if "DMSO" in moas:
            moas.remove("DMSO")  
        self.dataset = BBBC021(root_path=root_path, moa=moas)
        self.images = []
        self.compound_labels = []
        
        print(f"Loading and resizing {len(self.dataset)} images...")
        
        # Collect all images and optionally compound labels
        for i in range(len(self.dataset)):
            image, metadata = self.dataset[i]
            if metadata.compound.moa != 'null':
                # Convert to tensor if needed
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                
                # Apply resize to the tensor
                resized_image = self.resize_transform(image)
                self.images.append(resized_image)
                
                if self.compound_aware:
                    self.compound_labels.append(metadata.compound.compound)
        
        mode_str = "compound-aware" if self.compound_aware else "vanilla"
        print(f"Loaded {len(self.images)} resized images for {mode_str} SimCLR training")
        print(f"Memory usage per image: {self.images[0].element_size() * self.images[0].nelement() / 1024**2:.2f} MB")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # Already resized and converted to tensor
        
        # Apply augmentations twice to get two views
        if self.transform:
            aug1 = self.transform(image)
            aug2 = self.transform(image)
        else:
            aug1 = image
            aug2 = image
        
        if self.compound_aware:
            compound_label = self.compound_labels[idx]
            return aug1, aug2, compound_label
        else:
            return aug1, aug2


def contrastive_loss_vanilla(z1, z2, temperature=0.5):
    """
    Standard SimCLR NT-Xent loss for vanilla training.
    No labels needed - positive pairs are augmentations of the same image.
    """
    batch_size = z1.size(0)
    
    # Normalize features
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate all projections
    z = torch.cat([z1, z2], dim=0)  # Shape: (2*batch_size, projection_dim)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(z, z.T) / temperature
    
    # Create mask to remove self-similarities
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
    
    # Create positive pair labels
    # For batch_size=N: [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size), 
        torch.arange(batch_size)
    ], dim=0).to(z.device)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, positives)
    
    return loss


def contrastive_loss_vanilla_compound_aware(z1, z2, compound_labels, temperature=0.5):
    """
    Compound-aware SimCLR NT-Xent loss for vanilla training.
    Excludes same-compound pairs from being treated as negatives.
    This is my go at a less agressive WS version compared to the
    simclr_ws.py version as i couldn't believe our Labels would 
    not help at all :D.
    
    Args:
        z1, z2: Projected features from positive pairs (augmentations of same image).
        compound_labels: Compound labels for each image in the batch.
        temperature: Temperature parameter for softmax.
    """
    batch_size = z1.size(0)
    
    # Normalize features
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate all projections
    z = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(z, z.T) / temperature
    
    # Create positive pair labels (vanilla SimCLR style)
    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size), 
        torch.arange(batch_size)
    ], dim=0).to(z.device)
    
    # since we dont want to use same-compound negatives,
    # we need to create a mask that excludes them, like 
    # in the WS version.
    unique_compounds = list(set(compound_labels))
    compound_to_idx = {compound: idx for idx, compound in enumerate(unique_compounds)}
    compound_indices = torch.tensor([compound_to_idx[compound] for compound in compound_labels], 
                                   device=z.device)
    
    # Concatenate compound labels
    all_compound_indices = torch.cat([compound_indices, compound_indices])
    
    # Create a mask where (i, j) is True if images i and j are from the same compound
    same_compound_mask = (all_compound_indices.unsqueeze(0) == all_compound_indices.unsqueeze(1))
    
    # Create positive pair mask
    positive_pair_mask = torch.zeros_like(same_compound_mask)
    positive_pair_mask.scatter_(1, positives.unsqueeze(1), 1)
    
    # Final mask: mask out same-compound pairs that are NOT the designated positive pair
    final_mask = same_compound_mask & ~positive_pair_mask.bool()
    
    # Apply the mask to the similarity matrix
    similarity_matrix = similarity_matrix.masked_fill(final_mask, -float('inf'))
    
    # Mask out self-similarities (diagonal)
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity_matrix = similarity_matrix.masked_fill(self_mask, -float('inf'))
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, positives)
    
    return loss
