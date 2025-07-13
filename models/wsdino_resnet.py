import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from models.load_model import load_pretrained_model, ModelName

class BBBC021WeakLabelDataset(Dataset):
    """
    PyTorch-compatible dataset wrapper for BBBC021 using weak labels (compound IDs).

    Each sample consists of:
        - An image tensor (optionally transformed)
        - A weak label: the compound ID (as an index)
    
    Filters out samples with unknown MoA ('null').
    """
    def __init__(self, bbbc021, transform=None):
        """
        Args:
            bbbc021: An instance of pybbbc.BBBC021
            transform: A torchvision transform to apply to each image
        """
        self.bbbc021 = bbbc021
        self.transform = transform

        # Store only indices for samples with valid MoA
        self.valid_indices = [
            i for i, (_, meta) in enumerate(bbbc021)
            if meta.compound.moa != 'null'
        ]

        self.compound_to_idx = {
            compound: idx for idx, compound in enumerate(sorted(set(
                bbbc021[i][1].compound.compound for i in self.valid_indices
            )))
        }

    def __len__(self):
        #return len(self.valid_samples)
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img, meta = self.bbbc021[actual_idx]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()

        compound_id = meta.compound.compound
        weak_label = self.compound_to_idx[compound_id]
        moa = meta.compound.moa

        if self.transform:
            crops = self.transform(img)  # List of tensors
        else:
            crops = [img]

        return crops, weak_label, moa

class DINOProjectionHead(nn.Module):
    """
    DINO-style Projection Head for Self-Supervised Learning.

    This module implements a 3-layer MLP with:
        - Two hidden layers of size 2048 with GELU activations
        - An output layer of configurable dimension (default: 256) without activation
        - L2 normalization applied to the output
        - A weight-normalized linear layer as the final projection

    Args:
        in_dim (int): Input feature dimension (e.g., 2048 for ResNet-50).
        hidden_dim (int): Hidden layer dimension (default: 2048).
        proj_dim (int): Output projection dimension (default: 256).
    """
    def __init__(self, in_dim, proj_dim=256, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        # Optional weight-normed layer (as in DINO paper)
        self.last_layer = nn.utils.weight_norm(nn.Linear(proj_dim, proj_dim, bias=False))
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)  # L2 normalization
        return self.last_layer(x)

def get_resnet50(num_classes=None, use_projection_head=True, proj_dim=256, model_type="base_resnet"):
    """
    Loads a ResNet-50 backbone with the final layer replaced for classification.
    Optionally adds a DINO-style projection head.

    Args:
        num_classes: Number of output classes (e.g., number of unique MoAs, used only if projection head is False)
        use_projection_head: If True, attach DINO-style projection head
        proj_dim: Output dimension of projection head
        model_type: should be "base_resnet" or "wsdino"

    Returns:
        A torch.nn.Module (ResNet-50 with custom head)
    """
    backbone = load_pretrained_model(model_type)
    in_dim = backbone.fc.in_features

    if use_projection_head:
        backbone.fc = DINOProjectionHead(in_dim, proj_dim=proj_dim)
    else:
        assert num_classes is not None, "You must provide num_classes if not using projection head"
        backbone.fc = nn.Linear(in_dim, num_classes)

    return backbone

def dino_loss(student_out, teacher_out, temp=0.07):
    """
    Computes the DINO distillation loss between student and teacher outputs.

    Args:
        student_out: Logits from the student network
        teacher_out: Logits from the teacher network (detached)
        temp: Temperature scaling parameter

    Returns:
        KL divergence loss between student and teacher probability distributions
    """
    s = F.log_softmax(student_out / temp, dim=1)
    t = F.softmax(teacher_out / temp, dim=1).detach()
    return F.kl_div(s, t, reduction='batchmean')

def update_teacher(student, teacher, m=0.996):
    """
    Updates teacher weights using exponential moving average of student weights.

    Args:
        student: Student model (nn.Module)
        teacher: Teacher model (nn.Module)
        m: Momentum factor (closer to 1 = slower update)

    Returns:
        None (teacher model is updated in place)
    """
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data = pt.data * m + ps.data * (1. - m)

class MultiCropTransform:
    """
    Generate 2 global crops and N local crops from the same image.
    """
    def __init__(self, global_transform, local_transform, num_local_crops):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.num_local_crops = num_local_crops

    def __call__(self, x):
        crops = [self.global_transform(x) for _ in range(2)]
        crops += [self.local_transform(x) for _ in range(self.num_local_crops)]
        return crops