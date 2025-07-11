import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import gc

from models.load_model import load_pretrained_model
from models.simclr_vanilla import (SimCLRModel, 
                                   SimCLRVanillaDataset,
                                   contrastive_loss_vanilla,
                                   contrastive_loss_vanilla_compound_aware)

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
    # Note: Resize is now handled in the dataset initialization
    # We only apply augmentations here (no resize needed)
    simclr_transform = transforms.Compose([
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
        num_workers=min(4, os.cpu_count()),
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
    
    gc.collect()
    torch.cuda.empty_cache()
    model = train_simclr_vanilla(
        root_path="/scratch/cv-course2025/group8",
        epochs=200,
        batch_size=512,
        learning_rate=0.0006,
        temperature=0.3,
        projection_dim=128,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_every=100,
        compound_aware=True
    )
