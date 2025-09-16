# Cancer-Type-Aware Multimodal Survival Prediction

A robust framework for multimodal cancer survival prediction that operates reliably with incomplete data across diverse healthcare environments.

## Features

- **Adaptive Gated Fusion**: Handles arbitrary missing-modality scenarios without imputation
- **Cancer-Type-Aware Architecture**: Combines shared multimodal encoders with cancer-specific prediction heads
- **Clinical Deployment Ready**: Maintains performance across institutions with incomplete multimodal data

## Quick Start

### 1. Environment Setup

```bash
# Create environment from configuration
conda env create -f tcga.yaml
conda activate tcga_env
```

### 2. Install Dependencies

```bash
# Install ScatterMoE (required for mixture-of-experts components)
pip install -e scattermoe/

# Verify installation
cd scattermoe/
PYTHONPATH=. pytest tests
```

### 3. Basic Usage

```python
from scattermoe.mlp import MLP
import torch.nn as nn

# Initialize multimodal survival prediction module
mlp = MLP(
    input_size=x_dim, 
    hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, 
    top_k=k
)

# Forward pass with missing modalities support
Y = mlp(
    X,         # input tensor [batch_size, feature_dim]
    k_weights, # top-k weights from router
    k_idxs     # top-k indices from router
)
```

## Project Structure

```
├── running_scripts/        # Execution scripts
│   ├── survival_prediction_mainmoe.sh       #training script
├── model/config/           # Model configurations
│   ├── surv_multimodal_mainmoe.yml    # Our settings
│   └── default.yml        # Default parameters
├── data/                   # Dataset directory
│   └── RNA_embedding/              # rna dataset
│   └── text_embeddings/            # text dataset
│   └── tcga-dataset/              # image dataset
└── scattermoe/            # ScatterMoE implementation
```

## Running Experiments
### Download data files from Google Drive
### Visit: https://drive.google.com/drive/folders/1cPX564Bj2jNXfWKmqzVJse56UB3vyXZU?usp=sharing
### Download the entire 'data' folder and place it in the current directory
### Single Cancer Type (Example)

```bash
# Train on BLCA dataset with all modalities
bash running_scripts/survival_prediction_mainmoe.sh

## Configuration

Model configurations are stored in `model/config/`


## Dataset

We provide the **BLCA (Bladder Urothelial Carcinoma)** dataset as an example:
- Histopathological images (WSI patches)
- RNA expression profiles (BulkRNABert embeddings)
- Clinical text (pathology reports)
- Survival outcomes with censorship information



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact:
- Yiran Song: song0760@umn.edu
- Mingquan Lin: lin01231@umn.edu
