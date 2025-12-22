import torch
from .detect import detect
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metric import roc_auc
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from .detect import Mahalanobis

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['font.size'] = 12

def train_model(x, model, dataset_name, criterion, cri_kwargs, epochs, optimizer, verbose):
    losses = []
    min_loss = float('inf')
    epoch_iter = iter(_ for _ in range(epochs))
    if verbose:
        epoch_iter = tqdm(list(epoch_iter))
    for _ in epoch_iter:
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(x=x, y=y, **cri_kwargs)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        losses.append(current_loss)
        if current_loss < min_loss:
            min_loss = current_loss

        if verbose:
            epoch_iter.set_postfix({'loss': '{0:.4f}'.format(loss)})
    
    return min_loss

def calculate_pf_auc(detection_map, ground_truth):
    """
    Calculate AUC(Pf,τ)
    """
    thresholds = np.linspace(detection_map.min(), detection_map.max(), 100)
    pf_values = []
    
    for threshold in thresholds:
        binary_detection = detection_map > threshold
        background_mask = ground_truth == 0
        false_alarms = np.logical_and(binary_detection == 1, background_mask)
        pf = false_alarms.sum() / background_mask.sum()
        pf_values.append(pf)
    
    pf_auc = np.trapz(pf_values, thresholds)
    return pf_auc

def calculate_all_metrics(dm: np.ndarray, gt: np.ndarray):
    rows, cols = gt.shape
    gt = gt.reshape(rows * cols)
    dm = dm.reshape(rows * cols)
    
    if dm.max() != dm.min():
        dm = (dm - dm.min()) / (dm.max() - dm.min())
    else:
        dm = np.zeros_like(dm)
    
    pf, pd, _ = metrics.roc_curve(gt, dm)
    auc_pd_pf = metrics.auc(pf, pd)
    
    thresholds = np.linspace(0, 1, 10000)
    pd_values = []
    pf_values = []
    
    for threshold in thresholds:
        binary_detection = dm > threshold
        target_mask = gt == 1
        background_mask = gt == 0
        
        pd = np.logical_and(binary_detection == 1, target_mask).sum() / max(target_mask.sum(), 1e-10)
        pf = np.logical_and(binary_detection == 1, background_mask).sum() / max(background_mask.sum(), 1e-10)
        
        pd_values.append(pd)
        pf_values.append(pf)
    
    auc_pf_tau = np.trapz(pf_values, thresholds)
    auc_pd_tau = np.trapz(pd_values, thresholds)
    
    return auc_pd_pf, auc_pf_tau, auc_pd_tau, pf_values, pd_values

def separation_training(x: torch.Tensor, gt: np.ndarray, model, loss, mask, optimizer,
                        epochs, output_iter, max_iter, verbose, dataset_name: str):
    """
    Separation training algorithm.
    """
    result_path = os.path.join('results', model.name)
    os.makedirs(result_path, exist_ok=True)

    min_loss = float('inf')
    best_model_state = None
    best_dm = None
    best_model_output = None
    best_metrics = None
    best_auc = 0
    history = []
    loss_list = []
    
    for i in range(1, max_iter + 1):
        if verbose:
            print('Iter {0}'.format(i))

        model_input = x 
        current_loss = train_model(
            model_input,
            model,
            dataset_name,
            loss,
            {'mask': mask},
            epochs,
            optimizer,
            verbose
        )
        loss_list.append(current_loss)
        
        with torch.no_grad():
            model_output = model(model_input)
            dm = detect(x, model_output)
            np_dm = dm.cpu().detach().numpy()
            
            fpr, tpr, auc = roc_auc(np_dm, gt)
            auc_pd_pf, auc_pf_tau, auc_pd_tau, pf_values, pd_values = calculate_all_metrics(np_dm, gt)
            pf_auc = calculate_pf_auc(np_dm, gt)

        if auc_pd_pf > best_auc:
            best_auc = auc_pd_pf
            best_dm = np_dm.copy()
            best_pd_values = pd_values.copy()
            best_pf_values = pf_values.copy()
            best_model_state = model.state_dict().copy()
            best_model_output = model_output.cpu().numpy()
            best_metrics = {
                'auc_pd_pf': auc_pd_pf,
                'auc_pf_tau': auc_pf_tau,
                'auc_pd_tau': auc_pd_tau,
                'pf_auc': pf_auc,
                'pf_values': pf_values.copy(),
                'pd_values': pd_values.copy()
            }
            
            if verbose:
                print(f'New best model at iteration {i} with loss: {min_loss:.4f}')
                print(f'AUC(PD,PF): {auc_pd_pf:.4f}, AUC(PF,τ): {auc_pf_tau:.4f}, AUC(PD,τ): {auc_pd_tau:.4f}')

        mask.update(dm.detach())
        history.append(auc)

    if best_model_state is not None:
        model_save_path = os.path.join(result_path, f'{dataset_name}_best_model.pth')
        torch.save(best_model_state, model_save_path)
        if verbose:
            print(f'Best model saved to {model_save_path}')

    return best_dm, history, best_metrics['pf_values']
