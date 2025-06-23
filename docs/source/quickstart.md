# Quick Start

## Dataset Preparation

1. Download the BBBC021 dataset
2. Place it in the `data/` directory
3. Configure the data paths (configuration system to be implemented)

## Training a Model

*Note: Training scripts are currently in development. The following commands are planned:*

### SimCLR Training

```bash
# To be implemented
python scripts/train.py model=simclr data=bbbc021 trainer.max_epochs=100
```

### WS-DINO Training

```bash
# To be implemented
python scripts/train.py model=ws_dino data=bbbc021 trainer.max_epochs=100
```

## Configuration

*Configuration system using Hydra is planned. Key configuration files will include:*

- `configs/model/`: Model architectures (to be implemented)
- `configs/data/`: Dataset configurations (to be implemented)
- `configs/trainer/`: Training parameters (to be implemented)
- `configs/optimizer/`: Optimizer settings (to be implemented)

## Example Usage

```python
# Example structure - modules to be implemented
# from data import BBBC021Dataset
# from models import SimCLRModel

# Load dataset
# dataset = BBBC021Dataset(root="data/bbbc021")

# Initialize model
# model = SimCLRModel(backbone="resnet50", projection_dim=128)

# Training loop implementation pending...
```
