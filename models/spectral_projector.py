# Module 3: Spectral Projector
# Reference: docs/project_develop_plan.md Section 4

import torch
import torch.nn as nn


class SpectralProjector(nn.Module):
    """
    Project frequency-domain complex representations + uncertainty
    into backbone-compatible embeddings (no iFFT).

    Missing path:  V_miss [B,F,M] + σ [B,F,M] → e_miss [B,T,M,d]
    Observed path: x_obs  [B,T,K]              → e_obs  [B,T,K,d]

    Then reassemble into h [B,T,N,d] in original variable order.
    """

    def __init__(self, N: int, F: int, T: int, d_model: int):
        """
        Args:
            N: total number of variables
            F: number of frequency bins (T // 2 + 1)
            T: input sequence length
            d_model: embedding dimension
        """
        super().__init__()
        self.N = N
        self.F = F
        self.T = T
        self.d_model = d_model

        # Observed variable path: time domain → embedding
        self.obs_embedding = nn.Linear(1, d_model)

        # Missing variable path: [real, imag, σ] → embedding
        self.freq_to_emb = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Frequency dim (F) → Time dim (T) mapping
        self.freq_to_time = nn.Linear(F, T)

        # Learnable uncertainty scaling
        self.sigma_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x_obs: torch.Tensor,
        V_miss: torch.Tensor,
        sigma: torch.Tensor,
        obs_idx: torch.Tensor,
        miss_idx: torch.Tensor,
    ) -> tuple:
        """
        Args:
            x_obs: [B, T, K] observed time series (normalized)
            V_miss: [B, F, M] complex estimated missing spectra
            sigma: [B, F, M] real uncertainty
            obs_idx: [K] int tensor
            miss_idx: [M] int tensor
        Returns:
            h: [B, T, N, d] unified embedding
            attn_bias: [B, N] attention bias (0 for obs, σ-based for miss)
        """
        B, T, K = x_obs.shape
        M = miss_idx.shape[0]
        d = self.d_model

        # --- Observed path ---
        e_obs = self.obs_embedding(x_obs.unsqueeze(-1))  # [B, T, K, d]

        # --- Missing path ---
        real_part = V_miss.real      # [B, F, M]
        imag_part = V_miss.imag      # [B, F, M]
        sigma_scaled = sigma * self.sigma_scale  # [B, F, M]

        freq_features = torch.stack([real_part, imag_part, sigma_scaled], dim=-1)  # [B, F, M, 3]
        e_freq = self.freq_to_emb(freq_features)  # [B, F, M, d]

        # Map frequency dim → time dim
        e_freq = e_freq.permute(0, 2, 3, 1)  # [B, M, d, F]
        e_time = self.freq_to_time(e_freq)     # [B, M, d, T]
        e_miss = e_time.permute(0, 3, 1, 2)   # [B, T, M, d]

        # --- Reassemble in original variable order ---
        h = torch.zeros(B, T, self.N, d, device=x_obs.device)
        h[:, :, obs_idx, :] = e_obs
        h[:, :, miss_idx, :] = e_miss

        # --- Attention bias ---
        attn_bias = torch.zeros(B, self.N, device=x_obs.device)
        attn_bias[:, miss_idx] = sigma.mean(dim=1)  # [B, M] freq-averaged σ

        return h, attn_bias
