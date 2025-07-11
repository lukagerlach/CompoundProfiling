import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pybbbc import BBBC021
from models.wsdino_resnet import (
    BBBC021WeakLabelDataset,
    get_resnet50,
    dino_loss,
    update_teacher
)

def train_wsdino():
    # Settings
    batch_size = 16
    epochs = 10
    lr = 1e-4
    momentum = 0.996
    temperature = 0.07
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    bbbc = BBBC021(root_path="/scratch/cv-course2025/group8/bbbc021") 
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    dataset = BBBC021WeakLabelDataset(bbbc, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_compounds = len(dataset.compound_to_idx)

    # Create models
    student = get_resnet50(num_classes=num_compounds).to(device)
    teacher = get_resnet50(num_classes=num_compounds).to(device)

    # get weights from file
    # student = get_resnet50(num_classes=num_compounds, model_type="wsdino").to(device)
    # teacher = get_resnet50(num_classes=num_compounds, model_type="wsdino".WSDINO).to(device)

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

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
            if isinstance(model, nn.DataParallel):
                backbone_state = model.module.backbone.state_dict()
            else:
                backbone_state = model.backbone.state_dict()
                
            save_path = os.path.join(save_dir, f"resnet50_simclr_epoch_{epoch+1}.pth")
            torch.save(backbone_state, save_path)
            print(f"Model saved to {save_path}")

    # Save model
    torch.save(student.state_dict(), f"/scratch/cv-course2025/group8/model_weights/resnet50_wsdino.pth")
    print("Model saved to resnet50_wsdino.pth")

if __name__ == "__main__":
    train()
