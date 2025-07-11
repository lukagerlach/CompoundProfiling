# CompoundProfiling

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lukagerlach.github.io/CompoundProfiling/)

Pharmacological compound prediction from cell images

This project implements two self-supervised learning approaches for predicting the mechanism of action (MOA) of pharmacological compounds on the [BBBC021 human cell imaging dataset](https://bbbc.broadinstitute.org/BBBC021).

## Setup

This is a research project, not a package. To get started:

1. Clone the repository:
```bash
git clone https://github.com/lukagerlach/CompoundProfiling.git
cd CompoundProfiling
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the test script to verify setup:
```bash
python test.py
```

## Usage

Run Python scripts directly from the project root:
```bash
python <script_name>.py
```

All modules can be imported directly since they're in the root directory.
