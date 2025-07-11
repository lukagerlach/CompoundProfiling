import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np

from pybbbc import BBBC021, constants
from models.load_model import load_pretrained_model


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
        
        moas = constants.MOA.copy()
        if "null" in moas:
            moas.remove("null")
            
        self.dataset = BBBC021(root_path=root_path, moa=moas)
        self.images = []
        self.compound_labels = []
        
        # Collect all images and optionally compound labels
        for i in range(len(self.dataset)):
            image, metadata = self.dataset[i]
            if metadata.compound.moa != 'null':
                self.images.append(image)
                if self.compound_aware:
                    self.compound_labels.append(metadata.compound.compound)
        
        mode_str = "compound-aware" if self.compound_aware else "vanilla"
        print(f"Loaded {len(self.images)} images for {mode_str} SimCLR training")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
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


def train_simclr_vanilla(
    root_path="/scratch/cv-course2025/group8",
    epochs=200,
    batch_size=256,
    learning_rate=0.0003,
    temperature=0.5,
    projection_dim=128,
    device=None,
    save_every=50,
    save_dir="/scratch/cv-course2025/group8/model_weights/vanilla",
    compound_aware=False
):
    """
    Train vanilla SimCLR model using two augmentations of the same image,
    optionally with compound-aware loss that excludes same-compound negatives.
    
    Args:
        root_path: Path to BBBC021 dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        temperature: Temperature parameter for contrastive loss
        projection_dim: Output dimension of projection head
        device: Device to train on
        save_every: Save model every N epochs
        save_dir: Directory to save model weights
        compound_aware: If True, uses compound-aware loss that excludes same-compound negatives
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mode_str = "compound-aware" if compound_aware else "vanilla"
    print(f"Training {mode_str} SimCLR on device: {device}")
    
    # Strong augmentations for SimCLR
    # As we are using the same images for both augmentations here,
    # we need to ensure that the augmentations are strong enough
    # to create diverse views.
    simclr_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = SimCLRVanillaDataset(root_path=root_path, transform=simclr_transform, compound_aware=compound_aware)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        drop_last=True
    )
    
    # Load pretrained ResNet50 and create SimCLR model
    backbone = load_pretrained_model("base_resnet")
    model = SimCLRModel(backbone, projection_dim=projection_dim)
    
    # Use DataParallel for multi-GPU training
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for training")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_data in progress_bar:
            if compound_aware:
                aug1, aug2, compound_labels = batch_data
            else:
                aug1, aug2 = batch_data
                compound_labels = None
                
            aug1, aug2 = aug1.to(device), aug2.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, z1 = model(aug1)  # Features and projections for augmentation 1
            _, z2 = model(aug2)  # Features and projections for augmentation 2
            
            # Compute contrastive loss
            if compound_aware:
                loss = contrastive_loss_vanilla_compound_aware(z1, z2, compound_labels, temperature=temperature)
            else:
                loss = contrastive_loss_vanilla(z1, z2, temperature=temperature)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
            })
        
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Feature collapse monitoring (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Test with small batch
                test_batch_data = next(iter(dataloader))
                if compound_aware:
                    aug1, _, _ = test_batch_data
                else:
                    aug1, _ = test_batch_data
                aug1 = aug1[:16].to(device)
                
                features1, _ = model(aug1)
                
                # Feature similarity check
                features1_norm = F.normalize(features1, dim=1)
                
                # Pairwise similarities within batch
                sim_matrix = torch.mm(features1_norm, features1_norm.t())
                off_diag = sim_matrix[torch.eye(sim_matrix.size(0), device=device) == 0]
                
                print(f"  Feature similarity check:")
                print(f"    Mean off-diagonal similarity: {off_diag.mean():.4f}")
                print(f"    Std off-diagonal similarity: {off_diag.std():.4f}")
                
                if off_diag.mean() > 0.95:
                    print("    ⚠️ WARNING: High feature similarity detected!")
                
                # Check per-dimension standard deviation
                feature_std_by_dim = features1.std(dim=0)
                low_var_dims = (feature_std_by_dim < 0.01).sum().item()
                print(f"Dimensions with very low variance: {low_var_dims}/{features1.shape[1]}")
            
            model.train()
        
        # Save model every save_every epochs
        if (epoch + 1) % save_every == 0:
            if isinstance(model, nn.DataParallel):
                backbone_state = model.module.backbone.state_dict()
            else:
                backbone_state = model.backbone.state_dict()
                
            save_path = os.path.join(save_dir, f"resnet50_simclr_vanilla_epoch_{epoch+1}.pth")
            torch.save(backbone_state, save_path)
            print(f"Model saved to {save_path}")
    
    # Save final model
    if isinstance(model, nn.DataParallel):
        backbone_state = model.module.backbone.state_dict()
    else:
        backbone_state = model.backbone.state_dict()
        
    final_save_path = os.path.join(save_dir, "resnet50_simclr_vanilla.pth")
    torch.save(backbone_state, final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    return model


if __name__ == "__main__":
    # Train vanilla SimCLR model
    model = train_simclr_vanilla(
        root_path="/scratch/cv-course2025/group8",
        epochs=200,
        batch_size=256,
        learning_rate=0.0006,
        temperature=0.3,
        projection_dim=128,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_every=50,
        compound_aware=True
    )
