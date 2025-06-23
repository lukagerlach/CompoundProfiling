# Installation

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Git

## Setup

1. Clone the repository:
```bash
git clone https://github.com/vm-vh/CompoundProfiling.git
cd CompoundProfiling
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -r requirements.txt
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Verify Installation

```python
import torch
from models import ResNetBackbone

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```
