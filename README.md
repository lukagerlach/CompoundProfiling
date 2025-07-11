# CompoundProfiling

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lukagerlach.github.io/CompoundProfiling/)

*Predicting compound mechanisms of action from cellular imaging data*

This project explores self-supervised learning approaches for predicting the mechanism of action (MOA) of pharmacological compounds using microscopy images from the [BBBC021 dataset](https://bbbc.broadinstitute.org/BBBC021). We implement and compare three different representation learning methods to predict MOAs.

## Overview

The BBBC021 dataset contains fluorescence microscopy images of human cells treated with various compounds at different concentrations. Each image captures three cellular components: actin filaments, microtubules, and nuclei. Our goal is to learn meaningful representations that can predict a compound's mechanism of action based solely on the induced morphological changes.

### Approaches

**1. Baseline ResNet-50**
- Standard ImageNet-pretrained ResNet-50 for feature extraction
- Provides a baseline for comparison with self-supervised methods

**2. SimCLR (Self-Supervised Contrastive Learning)**
- *Vanilla SimCLR*: Creates positive pairs from augmented versions of the same image
- *Weakly-Supervised SimCLR*: Uses compound labels to form positive pairs from different images of the same compound, while treating different compounds as negatives

**3. DINO (Self-Distillation)**
- Weakly-supervised adaptation using compound labels
- Student-teacher architecture with exponential moving average updates

### Key Features
- **2 SSL Algortihm iImplementations** to predict compound MOAs
- **Typical Variation Normalization (TVN)**: Removes systematic noise by normalizing against DMSO control samples
- **Comprehensive Evaluation**: 1-nearest neighbor classification with multiple distance metrics
- **Visualization Pipeline**: t-SNE and UMAP embeddings for qualitative analysis
- **Jupyter Notebooks** for easy experimenting and exploration
- **Documentation** that looks very good lol

## Dataset

The BBBC021 dataset consists of:
- 103 distinct compounds with known mechanisms of action
- Multiple concentrations per compound (typically 8 concentrations)
- 4 replicate images per treatment condition
- 3-channel fluorescence images (1024×1280 pixels)
- 12 different mechanisms of action categories

## Installation
To get started:

1. **Clone the repository:**
```bash
git clone https://github.com/lukagerlach/CompoundProfiling.git
cd CompoundProfiling
```

2. **Set up Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download and preprocess the dataset:**

If you are on RAMSES you can just use our default location for data which is `/scratch/cv-course2025/group8/bbbc021`. If not, run:
```bash
python data/pybbbc_loader.py
```

## Usage

### Training Models

**Train vanilla SimCLR:**
```bash
python training/simclr_vanilla_train.py
```

**Train weakly-supervised SimCLR:**
```bash
python training/simclr_ws_train.py
```

**Train DINO model:**
```bash
python training/wsdino_resnet_train.py
```

### Feature Extraction

Extract features using a trained model:
```bash
python evaluation/extractor.py
```

### Evaluation

Evaluate model performance:
```bash
python evaluation/evaluator.py
```

### Visualization

Generate t-SNE/UMAP plots:
```bash
python evaluation/visualize_embeddings.py
```

## Project Structure

```
CompoundProfiling/
├── data/                    # Data loading and preprocessing
├── models/                  # Model architectures
├── training/               # Training scripts
├── evaluation/             # Evaluation and visualization tools
├── experiments/            # Experimental utilities (TVN, etc.)
├── notebooks/              # Jupyter notebooks for exploration
└── configs/                # Configuration files
```

## Results

TBA

## Dependencies

Core requirements:
- PyTorch >= 1.9.0
- torchvision
- scikit-learn
- numpy
- pandas
- matplotlib
- tqdm
- pybbbc (for dataset access)

See `requirements.txt` for complete dependency list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Useful Links

- [BBBC021 Dataset](https://bbbc.broadinstitute.org/BBBC021) - Original dataset homepage
- [pybbbc Documentation](https://pypi.org/project/pybbbc/) - Python package for BBBC021 access

- [SimCLR: A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709) - Original SimCLR paper
- [Self-Supervised Learning of Phenotypic Representations from Cell Images with Weak Labels](https://arxiv.org/abs/2209.07819) - WS-DINO method
- [GitHub Repository](https://github.com/lukagerlach/CompoundProfiling) - Source code
- [Project Documentation](https://lukagerlach.github.io/CompoundProfiling/) - Detailed docs