# CVFA Loss functions
# 2-term: L_total = α · L_forecast + β · L_spectral

import torch
import torch.nn as nn
from utils.metrics import masked_mae, masked_rmse


class CVFALoss(nn.Module):
    """
    Combined loss for CVFA training:
        L_total = α · L_forecast + β · L_spectral

    L_forecast:  MAE between predicted and true future values (VIDA-compatible)
    L_spectral:  amplitude MSE + phase cosine alignment
    """

    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forecast_loss(self, y_pred, y_true):
        """MAE forecast loss (VIDA-compatible, null_val=0.0)."""
        loss, _ = masked_mae(y_pred, y_true, null_val=0.0)
        return loss

    def spectral_alignment_loss(self, V_miss_hat, V_miss_true):
        """
        Spectral alignment loss: amplitude MSE + phase cosine similarity.

        Args:
            V_miss_hat:  [B, F, M] complex — estimated
            V_miss_true: [B, F, M] complex — ground truth
        """
        # Amplitude alignment
        amp_loss = nn.functional.mse_loss(V_miss_hat.abs(), V_miss_true.abs())

        # Phase alignment (cosine similarity of unit phasors)
        phase_pred = V_miss_hat / (V_miss_hat.abs() + 1e-8)
        phase_true = V_miss_true / (V_miss_true.abs() + 1e-8)
        phase_loss = 1 - (phase_pred * phase_true.conj()).real.mean()

        return amp_loss + phase_loss

    def forward(self, y_pred, y_true, V_miss_hat=None, V_miss_true=None):
        """
        Args:
            y_pred: predicted future values
            y_true: ground truth future values
            V_miss_hat: estimated missing spectra (None during forecaster-only training)
            V_miss_true: true missing spectra (None during inference)
        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        L_fc = self.forecast_loss(y_pred, y_true)
        loss_dict = {'forecast': L_fc.item()}

        total = self.alpha * L_fc

        if V_miss_hat is not None and V_miss_true is not None:
            L_sp = self.spectral_alignment_loss(V_miss_hat, V_miss_true)
            total = total + self.beta * L_sp
            loss_dict['spectral'] = L_sp.item()

        loss_dict['total'] = total.item()
        return total, loss_dict
