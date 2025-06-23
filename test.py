import subprocess
import os

import torch


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
command = ["nvidia-smi"]

print("PyTorch version:", torch.version)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")
    
    if device_count > 0:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU devices available")
else:
    print("CUDA not available")