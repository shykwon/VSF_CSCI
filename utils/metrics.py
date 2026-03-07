# Evaluation metrics — aligned with VIDA protocol
# Reference: vida-vsf/VIDA/util.py

import torch
import numpy as np


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss), torch.mean(loss, dim=1)


def masked_rmse(preds, labels, null_val=np.nan):
    mse_loss, per_instance = masked_mse(preds=preds, labels=labels, null_val=null_val)
    return torch.sqrt(mse_loss), torch.sqrt(per_instance)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss), torch.mean(loss, dim=1)


def metric(pred, real):
    """Compute MAE and RMSE (VIDA-compatible)."""
    mae = masked_mae(pred, real, 0.0)[0].item()
    rmse = masked_rmse(pred, real, 0.0)[0].item()
    return mae, rmse
