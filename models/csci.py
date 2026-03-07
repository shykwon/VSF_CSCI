# CSCI: Cross-Spectral Coherence Imputation — Full Model
# Reference: docs/project_develop_plan.md Section 7

import torch
import torch.nn as nn

from models.spectral_encoder import SpectralEncoder
from models.cross_spectral_estimator import CrossSpectralEstimator
from models.spectral_projector import SpectralProjector
from models.forecast_head import ForecastHead


class CSCI(nn.Module):
    """
    Full CSCI pipeline:
      1. SpectralEncoder: x_obs → V_obs (FFT)
      2. CrossSpectralEstimator: V_obs → V_miss_hat + σ (Wiener filter)
      3. SpectralProjector: V_obs, V_miss + σ → h [B,T,N,d] (embedding)
      4. ForecastHead: h → [B, in_dim, N, T] (forecaster-ready)

    Two head modes for ablation (H2 hypothesis):
      - 'embedding': h → Linear projection → forecaster  (CSCI 핵심 경로)
      - 'timeseries': V_miss → iFFT → 시계열 복원 → forecaster  (VIDA-style ablation)
    """

    def __init__(self, args):
        super().__init__()
        N = args.num_nodes
        T = args.seq_in_len
        F = T // 2 + 1
        d_model = args.d_model
        in_dim = args.in_dim
        lambda_reg = getattr(args, 'lambda_reg', 1e-3)
        head_mode = getattr(args, 'head_mode', 'embedding')
        self.n_rounds = getattr(args, 'n_rounds', 1)
        self.coherence_threshold = getattr(args, 'coherence_threshold', 0.3)
        self.threshold_decay = getattr(args, 'threshold_decay', 0.1)

        self.N = N
        self.T = T
        self.in_dim = in_dim
        self.head_mode = head_mode

        # Module 1: Spectral Encoder
        self.spectral_encoder = SpectralEncoder(T)

        # Module 2: Cross-Spectral Estimator
        self.cs_estimator = CrossSpectralEstimator(N, F, lambda_reg)

        # Module 3: Spectral Projector (always built, used in embedding mode)
        self.spectral_projector = SpectralProjector(N, F, T, d_model)

        # Module 4: Forecast Head (always outputs 1 channel — value only)
        # Extra channels (e.g. time_of_day) bypass CSCI and are concatenated after
        self.forecast_head = ForecastHead(
            mode=head_mode, N=N, T=T, d_model=d_model, in_dim=1,
        )

    def forward(
        self,
        x_full: torch.Tensor,
        obs_idx: torch.Tensor,
        miss_idx: torch.Tensor,
    ) -> tuple:
        """
        Args:
            x_full: [B, in_dim, N, T] full input (unobserved positions zeroed out)
            obs_idx: [K] int tensor — observed variable indices
            miss_idx: [M] int tensor — missing variable indices
        Returns:
            fc_input: [B, in_dim, N, T] — forecaster-ready input
            attn_bias: [B, N] — attention bias (σ-based, embedding mode only)
            V_miss_hat: [B, F, M] — estimated missing spectra (for loss)
            sigma: [B, F, M] — uncertainty (for loss)
        """
        B = x_full.shape[0]

        # Extract observed time series: [B, in_dim, N, T] → [B, T, K]
        x_obs = x_full[:, 0, obs_idx, :].transpose(1, 2)  # [B, T, K]

        # Step 1: FFT on observed variables
        V_obs = self.spectral_encoder(x_obs)  # [B, F, K]

        # Step 2: Wiener filter estimation (with optional hierarchical propagation)
        if self.n_rounds > 1:
            V_miss_hat, sigma = self.cs_estimator.forward_hierarchical(
                V_obs, obs_idx, miss_idx,
                n_rounds=self.n_rounds,
                base_threshold=self.coherence_threshold,
                threshold_decay=self.threshold_decay,
            )
        else:
            V_miss_hat, sigma = self.cs_estimator(V_obs, obs_idx, miss_idx)
        # V_miss_hat: [B, F, M], sigma: [B, F, M]

        # Step 3 & 4: diverge based on head mode
        if self.head_mode == 'embedding':
            # CSCI path: Spectral Projector → embedding → projection → forecaster input
            h, attn_bias = self.spectral_projector(
                x_obs, V_miss_hat, sigma, obs_idx, miss_idx
            )  # h: [B, T, N, d_model]
            fc_input = self.forecast_head(h=h)  # [B, in_dim, N, T]

            # σ-based confidence gating: high uncertainty → attenuated input
            # attn_bias [B, N]: 0 for observed, σ_mean for missing
            # exp(-x): x=0 → 1.0 (observed preserved), x>0 → <1.0 (missing attenuated)
            confidence = torch.exp(-attn_bias).unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            fc_input = fc_input * confidence

        else:  # 'timeseries' — ablation (VIDA-style iFFT)
            # iFFT path: V_miss → time series reconstruction → forecaster input
            fc_input = self.forecast_head(
                V_miss=V_miss_hat, x_obs=x_obs,
                obs_idx=obs_idx, miss_idx=miss_idx,
            )  # [B, in_dim, N, T]
            attn_bias = None

        # Bypass: concat extra channels (e.g. time_of_day) that don't go through CSCI
        # These are "always known" features — no imputation needed
        if self.in_dim > 1:
            extra_channels = x_full[:, 1:, :, :]  # [B, in_dim-1, N, T]
            fc_input = torch.cat([fc_input, extra_channels], dim=1)  # [B, in_dim, N, T]

        return fc_input, attn_bias, V_miss_hat, sigma
