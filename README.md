# TransGCF
TransGCF: A Unified Spatial-Spectral-Frequency Framework for Robust Hyperspectral Anomaly Detection
DOI: 10.1109/JSTARS.2026.3660283| (https://ieeexplore.ieee.org/document/11370489)
This repository contains the implementation of **TransGCF**, a framework for hyperspectral anomaly detection. 
## Framework Overview

TransGCF integrates three key components through gating networks:
- **MS_GELRSA**: Multi-Scale Graph-Enhanced Local Region Self-Attention
- **HFEB**: High-Frequency Enhancement Block
- **GCNBranch**: Graph Convolutional Network branch for spectral processing


## Installation

### Requirements

- Python 3.7+
- PyTorch 1.13+
- CUDA (for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset

This release includes support for the **abu-urban-4** dataset. The dataset file (`abu-urban-4.mat`) should be placed in the `dataset/abuu4/` directory.

The dataset should contain:
- `data`: Hyperspectral image data (H × W × Bands)
- `map`: Ground truth anomaly map (H × W)

## Usage

### Training

To train the TransGCF model on the abu-urban-4 dataset:

```bash
python main.py
```

### Configuration

You can modify the hyperparameters in `main.py`:

```python
lmda = 1e-3          # Regularization parameter
num_bs = 64          # Number of selected bands
lr = 1e-3            # Learning rate
epochs = 10          # Number of epochs per iteration
output_iter = 15     # Output iteration (early stopping)
max_iter = 30        # Maximum iterations
data_norm = True     # Whether to normalize data
```

## Results

After training, results will be saved in the `results/TransGCF/` directory:
- `{dataset_name}_roc.pdf`: ROC curve
- `{dataset_name}_metrics.csv`: Detailed metrics (PD, PF values)
- `{dataset_name}_fpr_data.csv`: FPR and TPR data for ROC curve

## Model Architecture

The TransGCF model is defined in `TransGCF.py`, which contains all necessary modules:

- **TransGCF**: Main model class implementing the three-layer gating fusion structure
- **MS_GELRSA**: Multi-Scale Graph-Enhanced Local Region Self-Attention
- **HFEB**: High-Frequency Enhancement Block
- **GCNBranch**: Graph Convolutional Network branch for spectral processing
- **GatingNetwork**: Gating fusion network

## Acknowledgments

This work is built upon the excellent foundation provided by:

- **MSNet**: Self-Supervised Multiscale Network with Enhanced Separation Training for Hyperspectral Anomaly Detection
  - Paper: [IEEE Transactions on Geoscience and Remote Sensing (TGRS)](https://ieeexplore.ieee.org/document/10551851)
  - Official Implementation: [https://github.com/enter-i-username/MSNet](https://github.com/enter-i-username/MSNet)
## Citation
If you find this work useful in your research, please cite our paper:
J. Xu et al., "TransGCF: A Unified Spatial-Spectral-Frequency Framework for Robust Hyperspectral Anomaly Detection," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2026.3660283
We gratefully acknowledge the authors of MSNet for their open-source implementation, which provided valuable insights and code structure for the development of TransGCF.

