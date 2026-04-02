# CVFA: Cross-Variate Frequency Attention for Variable Subset Forecasting
# Simplified 3-module pipeline (Tokenizer + CVFA + iFFT Reconstruction)

import torch
import torch.nn as nn

from models.spectral_encoder import SpectralEncoder
from models.cross_spectral_estimator import CrossSpectralEstimator


class CVFA(nn.Module):
    """
    CVFA pipeline (3 modules):
      1. Frequency Tokenizer: x_obs → V_obs (rFFT)
      2. Cross-Variate Frequency Attention: V_obs → V_miss_hat (Wiener filter)
      3. iFFT Reconstruction: [V_obs; V_miss_hat] → x_full_hat → backbone

    Design principles (vs prior CSCI):
      - iFFT로 시계열 원복 (no embedding injection)
      - 불확실성(σ) 제거 (모니터링만)
      - 백본 수정 없음 (1ch 그대로)
      - Loss 2-term (forecast + spectral)
    """

    def __init__(self, args):
        super().__init__()
        N = args.num_nodes
        T = args.seq_in_len
        F = T // 2 + 1
        in_dim = args.in_dim
        lambda_reg = getattr(args, 'lambda_reg', 0.1)
        s_rank = getattr(args, 's_rank', 16)
        self.n_rounds = getattr(args, 'n_rounds', 1)
        self.coherence_threshold = getattr(args, 'coherence_threshold', 0.3)
        self.threshold_decay = getattr(args, 'threshold_decay', 0.1)

        self.N = N
        self.T = T
        self.in_dim = in_dim
        # Extra channels (e.g. time_of_day) that bypass CVFA
        self.n_extra = in_dim - 1  # 0 for SOLAR/ECG/TRAFFIC, 1 for METR-LA

        # Module 1: Frequency Tokenizer (rFFT)
        self.spectral_encoder = SpectralEncoder(T)

        # Module 2: Cross-Variate Frequency Attention (Wiener filter)
        self.cs_estimator = CrossSpectralEstimator(N, F, lambda_reg, rank=s_rank)

    def forward(
        self,
        x_full: torch.Tensor,
        obs_idx: torch.Tensor,
        miss_idx: torch.Tensor,
        x_full_unmasked: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            x_full: [B, in_dim, N, T] masked input (unobserved positions zeroed out)
            obs_idx: [K] int tensor — observed variable indices
            miss_idx: [M] int tensor — missing variable indices
            x_full_unmasked: [B, in_dim, N, T] original unmasked input (for tod bypass)
        Returns:
            fc_input: [B, in_dim, N, T] — forecaster-ready input (backbone 수정 없음)
            V_miss_hat: [B, F, M] — estimated missing spectra (for spectral loss)
        """
        B = x_full.shape[0]

        # Extract observed time series: [B, in_dim, N, T] → [B, T, K]
        x_obs = x_full[:, 0, obs_idx, :].transpose(1, 2)  # [B, T, K]

        # Step 1: Frequency Tokenizer (rFFT on observed variables)
        V_obs = self.spectral_encoder(x_obs)  # [B, F, K]

        # Step 2: Cross-Variate Frequency Attention (Wiener filter)
        if self.n_rounds > 1:
            V_miss_hat, _sigma = self.cs_estimator.forward_hierarchical(
                V_obs, obs_idx, miss_idx,
                n_rounds=self.n_rounds,
                base_threshold=self.coherence_threshold,
                threshold_decay=self.threshold_decay,
            )
        else:
            V_miss_hat, _sigma = self.cs_estimator(V_obs, obs_idx, miss_idx)
        # V_miss_hat: [B, F, M]

        # Step 3: iFFT Reconstruction → full time series
        x_miss_hat = torch.fft.irfft(V_miss_hat, n=self.T, dim=1, norm='ortho')  # [B, T, M]

        # Reassemble full time series [B, T, N]
        x_recon = torch.zeros(B, self.T, self.N, device=x_full.device)
        x_recon[:, :, obs_idx] = x_obs                  # observed: original values
        x_recon[:, :, miss_idx] = x_miss_hat             # missing: iFFT reconstruction

        # [B, T, N] → [B, 1, N, T] (backbone-compatible format)
        fc_input = x_recon.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, N, T]

        # Bypass: concat extra channels (e.g. time_of_day) — not through CVFA
        if self.n_extra > 0:
            tod_source = x_full_unmasked if x_full_unmasked is not None else x_full
            extra_channels = tod_source[:, 1:, :, :]  # [B, n_extra, N, T]
            fc_input = torch.cat([fc_input, extra_channels], dim=1)

        return fc_input, V_miss_hat
