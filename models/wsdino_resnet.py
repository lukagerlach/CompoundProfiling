import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
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
        #self.bbbc021 = bbbc021
        self.transform = transform

        # Filter out samples with unknown MoA
        self.valid_samples = [
            # (img, self.compound_to_idx[m[1][0]])
            # for img, m in bbbc021 if m[1][2] != 'null'
            (img, metadata) for img, metadata in bbbc021
            if metadata.compound.moa != 'null'
        ]

                # Create Compound → index mapping
        self.compound_to_idx = {
            compound: idx for idx, compound in enumerate(sorted(set(
                # m[1][0] for _, m in bbbc021 if m[1][2] != 'null'  # skip unknown MoAs
                meta.compound.compound for _, meta in self.valid_samples
            )))
        }

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        #img, weak_label = self.valid_samples[idx]
        #moa = self.bbbc021[idx][1][2]  # access full meta for eval
        img, meta = self.valid_samples[idx]
        compound_id = meta.compound.compound
        weak_label = self.compound_to_idx[compound_id]
        moa = meta.compound.moa
        if self.transform:
            img = self.transform(img)
        return img, weak_label, moa

def get_resnet50(num_classes, model_type="base_resnet"):
    """
    Loads a ResNet-50 backbone with the final layer replaced for classification.

    Args:
        num_classes: Number of output classes (e.g., number of unique MoAs)
        model_type: should be "base_resnet" or "wsdino"

    Returns:
        A torch.nn.Module (ResNet-50 with custom head)
    """
    model = load_pretrained_model(model_type)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
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