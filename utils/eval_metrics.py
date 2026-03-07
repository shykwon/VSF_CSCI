# Extended evaluation metrics for CSCI paper
# CRPS, spectral metrics, missing rate sensitivity

import torch
import numpy as np


def crps_gaussian(mu, sigma, y):
    """
    Continuous Ranked Probability Score for Gaussian predictive distribution.

    Measures σ calibration quality: lower = better calibrated uncertainty.
    Used to validate Contribution 3 (theory-based σ quantification).

    Args:
        mu: [B, F, M] predicted mean (V_miss_hat amplitude)
        sigma: [B, F, M] predicted std (uncertainty)
        y: [B, F, M] ground truth (V_miss_true amplitude)
    Returns:
        scalar: mean CRPS
    """
    sigma = torch.clamp(sigma, min=1e-6)
    z = (y - mu) / sigma
    # CRPS = σ * (z * (2Φ(z) - 1) + 2φ(z) - 1/√π)
    phi_z = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)  # PDF
    Phi_z = 0.5 * (1 + torch.erf(z / np.sqrt(2)))           # CDF
    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    return crps.mean().item()


def spectral_amplitude_mse(V_hat, V_true):
    """
    MSE between predicted and true spectral amplitudes.
    Validates H1 hypothesis (cross-spectral estimation quality).

    Args:
        V_hat:  [B, F, M] complex — estimated missing spectra
        V_true: [B, F, M] complex — ground truth missing spectra
    Returns:
        scalar: amplitude MSE
    """
    amp_hat = V_hat.abs()
    amp_true = V_true.abs()
    return torch.nn.functional.mse_loss(amp_hat, amp_true).item()


def spectral_phase_mae(V_hat, V_true):
    """
    MAE of phase angle difference between predicted and true spectra.
    Validates H1 hypothesis (phase recovery quality).

    Args:
        V_hat:  [B, F, M] complex
        V_true: [B, F, M] complex
    Returns:
        scalar: phase MAE in radians
    """
    phase_hat = torch.angle(V_hat)
    phase_true = torch.angle(V_true)
    # Wrap phase difference to [-π, π]
    diff = phase_hat - phase_true
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff.abs().mean().item()


def spectral_coherence(V_hat, V_true):
    """
    Mean magnitude-squared coherence between estimated and true spectra.
    Perfect estimation = 1.0, no correlation = 0.0.

    Args:
        V_hat:  [B, F, M] complex
        V_true: [B, F, M] complex
    Returns:
        scalar: mean coherence
    """
    # Cross-spectral density
    S_xy = (V_hat * V_true.conj()).mean(dim=0)  # [F, M]
    S_xx = (V_hat * V_hat.conj()).real.mean(dim=0)  # [F, M]
    S_yy = (V_true * V_true.conj()).real.mean(dim=0)  # [F, M]
    coh = (S_xy.abs() ** 2) / (S_xx * S_yy + 1e-8)
    return coh.mean().item()


def evaluate_spectral_metrics(V_miss_hat, V_miss_true, sigma):
    """
    Compute all spectral evaluation metrics at once.

    Args:
        V_miss_hat:  [B, F, M] complex
        V_miss_true: [B, F, M] complex
        sigma:       [B, F, M] real
    Returns:
        dict with all spectral metrics
    """
    results = {}
    results['amplitude_mse'] = spectral_amplitude_mse(V_miss_hat, V_miss_true)
    results['phase_mae'] = spectral_phase_mae(V_miss_hat, V_miss_true)
    results['coherence'] = spectral_coherence(V_miss_hat, V_miss_true)
    results['crps'] = crps_gaussian(
        V_miss_hat.abs(), sigma, V_miss_true.abs()
    )
    return results
