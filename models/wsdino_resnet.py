from collections import defaultdict
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
# from torchvision import transforms
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

        self.label_to_indices = defaultdict(list)
        for i in self.valid_indices:
            compound = bbbc021[i][1].compound.compound
            self.label_to_indices[compound].append(i)
        
        # assert all(len(idxs) > 1 for idxs in self.label_to_indices.values()), "All compounds must have at least two samples."

    def __len__(self):
        #return len(self.valid_samples)
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img1, meta1 = self.bbbc021[actual_idx]
        compound = meta1.compound.compound
        weak_label = self.compound_to_idx[compound]
        moa = meta1.compound.moa

        # Sample a different image from same compound
        candidates = self.label_to_indices[compound]
        alt_idx = actual_idx
        while alt_idx == actual_idx:
            alt_idx = random.choice(candidates)

        img2, _ = self.bbbc021[alt_idx]

        # Convert img1 to torch tensor and ensure 3 channels
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float()
        if img1.ndim == 2:
            img1 = img1.unsqueeze(0).repeat(3, 1, 1)
        elif img1.shape[0] == 1:
            img1 = img1.repeat(3, 1, 1)

        # Convert img2 to torch tensor and ensure 3 channels
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float()
        if img2.ndim == 2:
            img2 = img2.unsqueeze(0).repeat(3, 1, 1)
        elif img2.shape[0] == 1:
            img2 = img2.repeat(3, 1, 1)

        if self.transform:
            crops1 = self.transform(img1)  # teacher views
            crops2 = self.transform(img2)  # student views
        else:
            crops1, crops2 = [img1], [img2]

        return crops1, crops2, weak_label, moa



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

class SimpleDINOLoss(nn.Module):
    """
    DINO Loss: Self-distillation with no labels.
    
    This version uses a constant teacher temperature (default: 0.04) and no temperature warmup.
    """
    def __init__(self, out_dim, ncrops, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops

        # Running center for teacher output
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Computes the DINO cross-entropy loss between teacher and student outputs.

        Args:
            student_output (Tensor): Shape (batch_size * ncrops, out_dim)
            teacher_output (Tensor): Shape (batch_size * 2, out_dim) — from 2 global views

        Returns:
            Scalar loss value (Tensor)
        """
        student_out = student_output / self.student_temp
        student_chunks = student_out.chunk(self.ncrops)

        # Centering and sharpening teacher outputs
        #teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = F.softmax((teacher_output - self.center.to(teacher_output.device)) / self.teacher_temp, dim=-1)
        teacher_chunks = teacher_out.detach().chunk(2)  # only global views

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_chunks):
            for v in range(len(student_chunks)):
                if v == iq:
                    continue  # skip matching student and teacher on same view
                loss = torch.sum(-q * F.log_softmax(student_chunks[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update the center used to normalize teacher outputs using EMA.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center.to(self.center.device)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


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

def update_teacher(student, teacher, m=0.99):
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
