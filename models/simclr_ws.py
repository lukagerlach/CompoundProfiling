import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from collections import defaultdict
import random

from pybbbc import BBBC021, constants
from models.load_model import create_feature_extractor


class SimCLRProjectionHead(nn.Module):
    """Projection head for SimCLR"""
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=256):
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
        This is our approach to WS-SimCLR where we use 
        weak labels to compose positive pairs. In particular,
        we used the compound labels to create positive pairs,
        where each pair consists of two images from the same
        compound but from different wells or plates (to bring
        in some noise and prevent the model from learning
        plate/well specific features).All negative pairs are 
        images from different compounds.
    """
    def __init__(self, backbone_model, projection_dim=256):
        super().__init__()
        # Remove the final classification layer from backbone
        self.backbone = create_feature_extractor(backbone_model)
        
        # Add projection head
        self.projection_head = SimCLRProjectionHead(
            input_dim=2048, 
            output_dim=projection_dim
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.squeeze()  # Remove spatial dimensions
        
        # Project features (litttle model on top of backbone),
        # as SimCLR uses a projection head
        projections = self.projection_head(features)
        
        return features, projections


class SimCLRDataset(Dataset):
    """Dataset for SimCLR training with compound-based positive pairs"""
    
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        
        # Filter MOAs to exclude DMSO and null
        moas = constants.MOA.copy()
        moas.remove('DMSO')
        moas.remove('null')
        self.data = BBBC021(root_path=root_path, moa=moas)
        
        # Group images by compound
        self.compound_groups = defaultdict(list)
        self.valid_indices = []
        
        for i, (image, metadata) in enumerate(self.data):
            compound = metadata.compound.compound
            plate_well = f"{metadata.plate.plate}_{metadata.plate.well}"
            
            self.compound_groups[compound].append({
                'index': i,
                'plate_well': plate_well,
                'image': image,
                'metadata': metadata
            })
            self.valid_indices.append(i)
        
        # Filter compounds with at least 2 images (from any plate/well)
        filtered_groups = {}
        for compound, images in self.compound_groups.items():
            if len(images) >= 2:  # At least 2 images for positive pairs
                filtered_groups[compound] = images
        
        self.compound_groups = filtered_groups
        self.compounds = list(self.compound_groups.keys())
        self.compound_to_idx = {c: i for i, c in enumerate(self.compounds)}
        
        print(f"Loaded {len(self.valid_indices)} valid images from {len(self.compounds)} compounds")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Select a random compound
        compound = random.choice(self.compounds)
        compound_idx = self.compound_to_idx[compound]
        compound_images = self.compound_groups[compound]
        
        # Try to select two images from different wells/plates for positive pair
        plate_wells = list(set(img['plate_well'] for img in compound_images))
        if len(plate_wells) >= 2:
            # Prefer different plate/well combinations if available
            well1, well2 = random.sample(plate_wells, 2)
            pos1 = random.choice([img for img in compound_images if img['plate_well'] == well1])
            pos2 = random.choice([img for img in compound_images if img['plate_well'] == well2])
        else:
            # If only one plate/well or not enough diversity, just select any two images
            pos1, pos2 = random.sample(compound_images, 2)
        
        # Get images
        pos1_image = pos1['image']
        pos2_image = pos2['image']
        
        # Convert to tensors and apply transforms
        images = []
        for img in [pos1_image, pos2_image]:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        return images[0], images[1], compound_idx  # pos1, pos2, label


def contrastive_loss(z1, z2, labels, temperature=0.1):
    """
    This was our go at the NT-Xent loss function, which is the
    contrastive loss used in SimCLR. It computes the loss
    based on the similarity of projected features from positive pairs.
    We made a small change to the loss function so that images in the 
    same compound are ingored as negatives, so the mdoel would not try 
    to seperate them  in the feature space. Positive pairs are
    images from the same compound, but from different wells/plates.
    
    Args:
        z1, z2: Projected features from positive pairs.
        labels: Compound labels for each pair in the batch.
        temperature: Temperature parameter for softmax.
    """
    batch_size = z1.shape[0]
    
    # Normalize features
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate all projections
    z = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Create labels for positive pairs
    positive_pair_labels = torch.cat([torch.arange(batch_size, 2*batch_size), 
                                      torch.arange(batch_size)]).to(z.device)
    
    # Here we take care of masking out the same-compound pairs
    all_labels = torch.cat([labels, labels]).unsqueeze(0)
    
    # Create a mask where (i, j) is True if images i and j are from the same compound
    same_compound_mask = (all_labels == all_labels.t())
    
    # Mask out the main diagonal, we dont want to compare imageas with themselves
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    same_compound_mask.masked_fill_(self_mask, False)
    
    # The logits for positive pairs should not be masked out
    positive_pair_mask = torch.zeros_like(same_compound_mask)
    positive_pair_mask.scatter_(1, positive_pair_labels.unsqueeze(1), 1)
    
    # Final mask: mask out same-compound pairs that are NOT the designated positive pair
    final_mask = same_compound_mask & ~positive_pair_mask.bool()

    # Apply the mask to the similarity matrix
    sim_matrix.masked_fill_(final_mask, -float('inf'))
    
    # Mask out self-similarities (diagonal)
    sim_matrix.masked_fill_(self_mask, -float('inf'))
    
    # Compute loss
    loss = F.cross_entropy(sim_matrix, positive_pair_labels)
    
    return loss


class LARS(optim.Optimizer):
    """
        LARS optimizer implementation. Although this
        Optimizer is designed for large batch sizes,
        we still wanted to implement this since it
        is the optimizer used in the original
        SimCLR paper.
        
        However, given our computational resources,
        we will use AdamW instead, which is more
        suitable for our setup.
    """
    def __init__(self, params, lr=1.0, momentum=0.9, weight_decay=1e-4, eta=1e-3):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(p.grad.data)

                if param_norm != 0 and grad_norm != 0:
                    # Compute local learning rate
                    local_lr = eta * param_norm / (grad_norm + weight_decay * param_norm)
                    local_lr = min(local_lr, lr)
                else:
                    local_lr = lr

                # Apply weight decay
                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)

                # Apply momentum
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(p.grad.data)

                # Apply update
                p.data.add_(buf, alpha=-local_lr)

        return loss
