# CSCI Loss functions
# Reference: docs/project_develop_plan.md Section 6-3

import torch
import torch.nn as nn
from utils.metrics import masked_mae, masked_rmse


class CSCILoss(nn.Module):
    """
    Combined loss for CSCI training:
        L_total = α · L_forecast + β · L_spectral + γ · L_uncertainty

    L_forecast:    MAE between predicted and true future values (VIDA-compatible)
    L_spectral:    amplitude MSE + phase cosine alignment
    L_uncertainty: σ calibration against actual estimation error
    """

    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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

    def uncertainty_calibration_loss(self, sigma, V_miss_hat, V_miss_true):
        """
        Calibrate σ to match actual estimation error.
        Uses .detach() so uncertainty doesn't affect prediction gradient.

        Args:
            sigma:       [B, F, M] real
            V_miss_hat:  [B, F, M] complex
            V_miss_true: [B, F, M] complex
        """
        actual_error = (V_miss_hat - V_miss_true).abs().mean(dim=1)  # [B, M]
        sigma_mean = sigma.mean(dim=1)  # [B, M]
        return nn.functional.mse_loss(sigma_mean, actual_error.detach())

    def forward(self, y_pred, y_true, V_miss_hat=None, V_miss_true=None, sigma=None):
        """
        Args:
            y_pred: predicted future values
            y_true: ground truth future values
            V_miss_hat: estimated missing spectra (None during forecaster-only training)
            V_miss_true: true missing spectra (None during inference)
            sigma: uncertainty estimates
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

            if sigma is not None:
                L_uc = self.uncertainty_calibration_loss(sigma, V_miss_hat, V_miss_true)
                total = total + self.gamma * L_uc
                loss_dict['uncertainty'] = L_uc.item()

        loss_dict['total'] = total.item()
        return total, loss_dict
