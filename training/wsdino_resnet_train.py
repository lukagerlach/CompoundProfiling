import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
from pybbbc import BBBC021
from models.wsdino_resnet import (
    BBBC021WeakLabelDataset,
    MultiCropTransform,
    get_resnet50,
    SimpleDINOLoss,
    update_teacher
)

def train_wsdino(
    root_path="/scratch/cv-course2025/group8",
    epochs=200,
    batch_size=16,
    lr = 4 * 10**(-6),
    momentum = 0.99,
    temperature=0.04,
    proj_dim=128,
    save_every=50
):
    """
    Training WS-DINO with a ResNet50 backbone on the BBBC021 dataset.
    
    Args:
        root_path: Path to BBBC021 dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for optimizer
        momentum: 
        temperature: Temperature parameter for DINO-style loss
        projection_dim: Output dimension of projection head
        save_every: Save model every N epochs
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use DataParallel for multi-GPU training
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for training")

    # Load data
    #bbbc = BBBC021(root_path=os.path.join(root_path, "bbbc021"))
    bbbc = BBBC021(root_path)
    print("loaded data")

    global_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        #T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        #T.ToTensor(),
        #T.Resize((224, 224)),
        T.Normalize(mean=[0.485]*3, std=[0.229]*3)
    ])

    local_transform = T.Compose([
        T.RandomResizedCrop(96, scale=(0.05, 0.4)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        #T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.Normalize(mean=[0.485]*3, std=[0.229]*3)
    ])

    # Combine into a multi-crop transform
    transform = MultiCropTransform(global_transform, local_transform, num_local_crops=6)

    # Create dataset and dataloader
    dataset = BBBC021WeakLabelDataset(bbbc, transform=transform)
    print("made dataset")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(10, os.cpu_count()),
        pin_memory=True,
        drop_last=True
    )
    print("dataloader created")

    num_compounds = len(dataset.compound_to_idx)

    # Create models
    student = get_resnet50(num_classes=num_compounds, proj_dim=proj_dim, model_type="base_resnet")
    print("student initialized")
    teacher = get_resnet50(num_classes=num_compounds, proj_dim=proj_dim, model_type="base_resnet")
    print("teacher initialized")

    # get weights from file
    #student = get_resnet50(num_classes=num_compounds, proj_dim=proj_dim, model_type="wsdino")
    #teacher = get_resnet50(num_classes=num_compounds, proj_dim=proj_dim, model_type="wsdino")

    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        student = nn.DataParallel(student)
        teacher = nn.DataParallel(teacher)
    student = student.to(device)
    teacher = teacher.to(device)

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # Optimizer
    #optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=0.0,
        betas=(0.9, 0.999)
    )

    criterion = SimpleDINOLoss(
        out_dim=proj_dim,
        ncrops=8,               # 2 global + 6 local crops
        student_temp=0.1,
        teacher_temp=temperature,
        center_momentum=momentum
    )

    # Create save directory
    save_dir = os.path.join(root_path, "model_weights")
    os.makedirs(save_dir, exist_ok=True)

    print("starting training loop")
    # Training loop
    for epoch in range(epochs):
        student.train()
        total_loss = 0

        for crops1, crops2, _, _ in dataloader:
            crops1 = [c.to(device) for c in crops1]  # teacher views
            crops2 = [c.to(device) for c in crops2]  # student views

            #student_outputs = torch.cat([student(crop) for crop in crops2], dim=0)
            #with torch.no_grad():
            #    teacher_outputs = torch.cat([teacher(crop) for crop in crops1], dim=0)

            # Flatten all views into a single batch
            all_student_views = torch.cat(crops2, dim=0)  # shape [num_crops * B, C, H, W]
            all_teacher_views = torch.cat(crops1, dim=0)  # shape [2 * B, C, H, W]

            # Forward passes
            student_outputs = student(all_student_views)
            with torch.no_grad():
                teacher_outputs = teacher(all_teacher_views)


            # Compute DINO loss
            loss = criterion(student_outputs, teacher_outputs)

            # Backward & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher
            update_teacher(student, teacher, m=momentum)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model every save_every epochs
        if (epoch + 1) % save_every == 0 or epoch == 1:
            model_to_save = student.module if isinstance(student, nn.DataParallel) else student
            backbone_state = model_to_save.state_dict()

            save_path = os.path.join(save_dir, f"resnet50_wsdino_epoch_{epoch+1}.pth")
            torch.save(backbone_state, save_path)
            print(f"Model saved to {save_path}")

        torch.cuda.empty_cache()

    # Save model
    torch.save(student.state_dict(), os.path.join(save_dir, "resnet50_wsdino.pth"))
    print("Model saved to resnet50_wsdino.pth")

if __name__ == "__main__":
    train_wsdino(
        root_path="/scratch/cv-course2025/group8",
        epochs=100,
        batch_size=16,
        lr=4 * 10**(-6),
        momentum = 0.99,
        temperature=0.04,
        proj_dim=128,
        save_every=50
    )
