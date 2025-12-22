import torch
import torch.nn.functional as F
import numpy as np

def detect(x, decoder_outputs):
    if isinstance(decoder_outputs, (tuple, list)):
        y = decoder_outputs[0]
    else:
        y = decoder_outputs
    dm = (x - y).detach()
    dm = dm ** 2
    dm = dm.sum(2)

    return dm


def Mahalanobis(data):
    row, col, band = data.shape
    
    data = data.reshape(row * col, band)
    
    mean_vector = torch.mean(data, dim=0)
    
    mean_matrix = mean_vector.repeat(row * col, 1)
    
    re_matrix = data - mean_matrix
    
    matrix = torch.matmul(re_matrix.T, re_matrix) / (row * col - 1)
    
    variance_covariance = torch.linalg.pinv(matrix)
    
    distances = torch.zeros(row * col, 1, device=data.device)
    
    for i in range(row * col):
        re_array = re_matrix[i]
        re_var = torch.matmul(re_array, variance_covariance)
        distances[i] = torch.matmul(re_var, re_array.T)
    
    distances = distances.reshape(row, col)
    
    return distances
