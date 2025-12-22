"""
TransGCF - Main Training Script
This script demonstrates how to train the TransGCF framework on the abu-urban-4 dataset.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.optim import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import abuu4
from TransGCF import TransGCF
import select_bands
import utils
import metric
from SeT import (
    calculate_all_metrics,
    TotalLoss,
    Mask,
    separation_training
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lmda = 1e-3
num_bs = 64
lr = 1e-3
epochs = 10
output_iter = 15
max_iter = 30
data_norm = True

dataset = abuu4
data, gt = dataset.get_data()
rows, cols, bands = data.shape
print(f'Processing dataset: {dataset.name}')
print(f'Data shape: {rows} x {cols} x {bands}')

band_idx = select_bands.OPBS(data, num_bs)
data_bs = data[:, :, band_idx]

if data_norm:
    data_bs = utils.ZScoreNorm().fit(data_bs).transform(data_bs)

net_kwargs = {'shape': (rows, cols, num_bs)}
model = TransGCF(**net_kwargs).to(device).float()
print(f'Model: {model.name}')

loss = TotalLoss(lmda, device)
mask = Mask((rows, cols), device)
optimizer = Adam(model.parameters(), lr=lr)

x_bs = torch.from_numpy(data_bs).to(device).float()

print('\nStarting training...')
pr_dm, history, history_pf = separation_training(
    x=x_bs,
    gt=gt,
    model=model,
    loss=loss,
    mask=mask,
    optimizer=optimizer,
    epochs=epochs,
    output_iter=output_iter,
    max_iter=max_iter,
    verbose=True,
    dataset_name=dataset.name
)

result_path = os.path.join('results', model.name)
os.makedirs(result_path, exist_ok=True)

fpr, tpr, pr_auc = metric.roc_auc(pr_dm, gt)
pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
    os.path.join(result_path, f'{dataset.name}_fpr_data.csv'), index=False)

plt.figure()
plt.plot(fpr, tpr, label=f'{model.name}: {pr_auc:.4f}', c='black', alpha=0.7)
plt.grid(alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title(f'ROC Curve - {dataset.name}')
plt.savefig(os.path.join(result_path, f'{dataset.name}_roc.pdf'))
plt.close()

auc_pd_pf, auc_pf_tau, auc_pd_tau, pf_values, pd_values = calculate_all_metrics(pr_dm, gt)

pd.DataFrame({
    'Threshold': np.linspace(0, 1, 10000),
    'Pd': pd_values,
    'Pf': pf_values
}).to_csv(os.path.join(result_path, f'{dataset.name}_metrics.csv'), index=False)

print(f'\n=== Results for {dataset.name} ===')
print(f'AUC(PD,PF): {auc_pd_pf:.4f}')
print(f'AUC(PF,τ): {auc_pf_tau:.4f}')
print(f'AUC(PD,τ): {auc_pd_tau:.4f}')
print(f'Final AUC: {history[output_iter - 1]:.4f}')
print(f'\nResults saved to: {result_path}')
