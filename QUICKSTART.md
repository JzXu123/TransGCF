# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset file is in place:
   - Place `abu-urban-4.mat` in `dataset/abuu4/` directory
   - The file should contain:
     - `data`: Hyperspectral image (H × W × Bands)
     - `map`: Ground truth anomaly map (H × W)

## Running the Code

Simply run:
```bash
python main.py
```

## Expected Output

The script will:
1. Load the abu-urban-4 dataset
2. Select bands using OPBS algorithm
3. Normalize the data
4. Train the TransGCF model using separation training
5. Save results to `results/TransGCF/`:
   - ROC curve plot
   - Metrics CSV file
   - FPR/TPR data

## Configuration

Edit hyperparameters in `main.py`:
- `num_bs`: Number of selected bands (default: 64)
- `lr`: Learning rate (default: 1e-3)
- `epochs`: Epochs per iteration (default: 10)
- `max_iter`: Maximum iterations (default: 30)
- `output_iter`: Early stopping iteration (default: 15)

## Troubleshooting

1. **CUDA out of memory**: Reduce `num_bs` or use CPU
2. **Dataset not found**: Ensure `abu-urban-4.mat` is in `dataset/abuu4/`
3. **Import errors**: Check that all dependencies are installed

