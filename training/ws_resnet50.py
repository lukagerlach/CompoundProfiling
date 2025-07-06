import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from pybbbc import BBBC021
from models.resnet_50_base import load_pretrained_model, MODEL_NAMES

class BBBC021TorchDataset(Dataset):
    """
    PyTorch-compatible dataset wrapper for the BBBC021 dataset using pybbbc.

    Each sample consists of:
        - An image tensor (optionally transformed)
        - A label corresponding to the Mechanism of Action (MoA) class index

    Filters out samples with unknown MoA ('null'), and assigns a unique index
    to each known MoA.
    """
    def __init__(self, bbbc021, transform=None):
        """
        Args:
            bbbc021: An instance of pybbbc.BBBC021
            transform: A torchvision transform to apply to each image
        """
        self.bbbc021 = bbbc021
        self.transform = transform

        # Create MoA → class index mapping (excluding 'null')
        self.moa_to_idx = {
            moa: idx for idx, moa in enumerate(sorted(set(
                m[1][2] for _, m in bbbc021 if m[1][2] != 'null'
            )))
        }

        # Build a list of valid (image, label) pairs
        self.valid_samples = [
            (img, self.moa_to_idx[m[1][2]])
            for img, m in bbbc021 if m[1][2] != 'null'
        ]

    def __len__(self):
        """Returns number of valid samples"""
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the sample

        Returns:
            (transformed image tensor, label index)
        """
        img, label = self.valid_samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_resnet50(num_classes, model_type=MODEL_NAMES.BASE_RESNET):
    """
    Loads a ResNet-50 backbone with the final layer replaced for classification.

    Args:
        num_classes: Number of output classes (e.g., number of unique MoAs)
        model_type: Enum specifying the type of pretrained model to load, must be MODEL_NAMES.BASE_RESNET or MODEL_NAMES.WSDINO

    Returns:
        A torch.nn.Module (ResNet-50 with custom head)
    """
    model = load_pretrained_model(model_type)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)  # replace head
    return model

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
