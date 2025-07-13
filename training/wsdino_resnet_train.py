import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
from pybbbc import BBBC021
from models.wsdino_resnet import (
    BBBC021WeakLabelDataset,
    get_resnet50,
    dino_loss,
    update_teacher
)

def train_wsdino(
    root_path="/scratch/cv-course2025/group8",
    epochs=200,
    batch_size=512,
    lr=0.0003,
    momentum = 0.996,
    temperature=0.1,
    # projection_dim=128,
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
    bbbc = BBBC021(root_path=os.path.join(root_path, "bbbc021"))

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = BBBC021WeakLabelDataset(bbbc, transform=transform)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(16, os.cpu_count()),
        pin_memory=True,
        drop_last=True
    )

    num_compounds = len(dataset.compound_to_idx)

    # Create models
    student = get_resnet50(num_classes=num_compounds)
    teacher = get_resnet50(num_classes=num_compounds)

    # get weights from file
    #student = get_resnet50(num_classes=num_compounds, model_type="wsdino")
    #teacher = get_resnet50(num_classes=num_compounds, model_type="wsdino")

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
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    # weight_decay=0.05, betas=(0.9, 0.999)

    # Create save directory
    save_dir = os.path.join(root_path, "model_weights")
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for images, _, _ in dataloader:  # ignore weak_label and moa
            images = images.to(device)

            student_out = student(images)
            with torch.no_grad():
                teacher_out = teacher(images)

            loss = dino_loss(student_out, teacher_out, temp=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_teacher(student, teacher, m=momentum)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model every save_every epochs
        if (epoch + 1) % save_every == 0:
            model_to_save = student.module if isinstance(student, nn.DataParallel) else student
            backbone_state = model_to_save.state_dict()

            save_path = os.path.join(save_dir, f"resnet50_wsdino_epoch_{epoch+1}.pth")
            torch.save(backbone_state, save_path)
            print(f"Model saved to {save_path}")

    # Save model
    torch.save(student.state_dict(), os.path.join(save_dir, "resnet50_wsdino.pth"))
    print("Model saved to resnet50_wsdino.pth")

if __name__ == "__main__":
    train_wsdino(
        root_path="/scratch/cv-course2025/group8",
        epochs=200,
        batch_size=512,
        lr=0.0003,
        momentum = 0.996,
        #projection_dim=128,
        save_every=50
    )
