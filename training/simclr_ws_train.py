import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from models.load_model import load_pretrained_model
from models.simclr_ws import (
    SimCLRModel,
    SimCLRDataset,
    contrastive_loss
)

def train_simclr(
    root_path="/scratch/cv-course2025/group8",
    epochs=200,
    batch_size=512,
    learning_rate=0.0003,
    temperature=0.1,
    projection_dim=128,
    device=None,
    save_every=20
):
    """
    Training wrapper for SimCLR model on BBBC021 dataset
    
    Args:
        root_path: Path to BBBC021 dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        temperature: Temperature parameter for contrastive loss
        projection_dim: Output dimension of projection head
        device: Device to train on
        save_every: Save model every N epochs
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transforms with reduced aggressiveness
    # Here we use a simple resize and normalize transform
    # As we are already using different images for positive pairs,
    # there is no need for any extra augmentations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = SimCLRDataset(root_path=root_path, transform=transform)
    
    # Use DataParallel for multi-GPU training
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for training")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(16, os.cpu_count()),
        pin_memory=True,
        drop_last=True
    )
    
    # Load pretrained ResNet50 and create SimCLR model
    backbone = load_pretrained_model("base_resnet")
    model = SimCLRModel(backbone, projection_dim=projection_dim)
    
    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    '''
    # LARS optimizer mit korrigierter Skalierung
    optimizer = LARS(
        model.parameters(), 
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-7,
        eta=1e-3
    )
    '''
    # Replace LARS with AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    # Learning rate scheduler
    # No need for scheduler for Adam
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create save directory
    save_dir = "/scratch/cv-course2025/group8/model_weights"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for pos1, pos2, labels in progress_bar:
            # Note: pos1, pos2 a re batches of images
            # labels are the compound indices for each image in the batch
            pos1, pos2, labels = pos1.to(device), pos2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass, we only need the projektions
            _, z1 = model(pos1)
            _, z2 = model(pos2)
            
            # Compute contrastive loss (on projections)
            loss = contrastive_loss(z1, z2, labels, temperature=temperature)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # some progress updates
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                # 'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model every save_every epochs
        if (epoch + 1) % save_every == 0:
            if isinstance(model, nn.DataParallel):
                backbone_state = model.module.backbone.state_dict()
            else:
                backbone_state = model.backbone.state_dict()
                
            save_path = os.path.join(save_dir, f"resnet50_simclr_epoch_{epoch+1}.pth")
            torch.save(backbone_state, save_path)
            print(f"Model saved to {save_path}")
    
    # Save final model
    if isinstance(model, nn.DataParallel):
        backbone_state = model.module.backbone.state_dict()
    else:
        backbone_state = model.backbone.state_dict()
        
    final_save_path = os.path.join(save_dir, "resnet50_simclr.pth")
    torch.save(backbone_state, final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    return model


if __name__ == "__main__":
    # Train SimCLR model
    model = train_simclr(
        root_path="/scratch/cv-course2025/group8",
        epochs=200,
        batch_size=512,
        learning_rate=0.1,
        temperature=0.1,
        projection_dim=128,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_every=100
    )
