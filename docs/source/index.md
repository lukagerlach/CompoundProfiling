# Compound Profiling Documentation

Welcome to the Compound Profiling project documentation!

This project implements self-supervised learning approaches for predicting compound Mechanism of Action (MOA) using the BBBC021 human cell imaging dataset.

## Overview

We benchmark two self-supervised learning algorithms by fine-tuning a ResNet-50 (pretrained on ImageNet):

- **SimCLR**: Contrastive SSL approach with weak labels
- **WS-DINO**: Non-contrastive SSL method with weak labels

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started:

readme
installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: API Reference:

api/index
```

```{toctree}
:maxdepth: 2
:caption: Research:

experiments
```

```{toctree}
:maxdepth: 1
:caption: Project Information:

authors
license
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
