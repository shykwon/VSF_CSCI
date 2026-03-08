# Forecast Head Adapter
# Bridges CSCI output (embedding dim) to forecaster input (time series dim).
# Also supports time-series path for ablation (iFFT reconstruction, like VIDA).

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """
    Adapts CSCI output to forecaster input format.

    Three modes:
      1. 'embedding': h [B, T, N, d_model] → permute → [B, d_model, N, T]
         - 백본의 start_conv/skip0 교체 필요.

      2. 'projected': miss 임베딩을 1채널로 projection + obs 원본 → [B, 1, N, T]
         - 백본 입력층 교체 불필요 → Stage 1 가중치 완전 재활용.
         - obs는 원본 시계열 그대로, miss는 학습된 projection.

      3. 'timeseries' (ablation): V_miss → iFFT → [B, 1, N, T]
         - VIDA-style. H2 가설 검증용.
    """

    def __init__(self, mode: str, N: int, T: int, d_model: int = 64, in_dim: int = 1):
        super().__init__()
        self.mode = mode
        self.N = N
        self.T = T
        self.d_model = d_model
        self.in_dim = in_dim

        if mode == 'embedding':
            pass
        elif mode == 'projected':
            # Learned projection: d_model embedding → 1-channel time series
            self.miss_proj = nn.Linear(d_model, 1)
        elif mode == 'timeseries':
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'embedding', 'projected', or 'timeseries'.")

    def forward(
        self,
        # Embedding mode inputs
        h: torch.Tensor = None,            # [B, T, N, d_model]
        # Timeseries mode inputs
        V_miss: torch.Tensor = None,       # [B, F, M] complex
        x_obs: torch.Tensor = None,        # [B, T, K] real
        obs_idx: torch.Tensor = None,      # [K]
        miss_idx: torch.Tensor = None,     # [M]
    ) -> torch.Tensor:
        """
        Returns:
            embedding mode: [B, d_model, N, T]
            timeseries mode: [B, 1, N, T]
        """
        if self.mode == 'embedding':
            return self._embedding_path(h)
        elif self.mode == 'projected':
            return self._projected_path(h, x_obs, obs_idx, miss_idx)
        else:
            return self._timeseries_path(V_miss, x_obs, obs_idx, miss_idx)

    def _embedding_path(self, h: torch.Tensor) -> torch.Tensor:
        """
        h [B, T, N, d_model] → permute → [B, d_model, N, T]
        No projection — full embedding passed to backbone.
        """
        return h.permute(0, 3, 2, 1)  # [B, d_model, N, T]

    def _projected_path(
        self,
        h: torch.Tensor,
        x_obs: torch.Tensor,
        obs_idx: torch.Tensor,
        miss_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Projected mode: miss embedding → Linear → 1ch + obs original → [B, 1, N, T]
        No backbone input layer replacement needed.

        h [B, T, N, d_model] — only miss positions used
        x_obs [B, T, K] — original observed time series
        """
        B = x_obs.shape[0]

        # Miss: project embedding to 1-channel
        h_miss = h[:, :, miss_idx, :]          # [B, T, M, d_model]
        x_miss_proj = self.miss_proj(h_miss).squeeze(-1)  # [B, T, M]

        # Reassemble: obs original + miss projected
        x_full = torch.zeros(B, self.T, self.N, device=x_obs.device)
        x_full[:, :, obs_idx.cpu()] = x_obs
        x_full[:, :, miss_idx.cpu()] = x_miss_proj

        # [B, T, N] → [B, 1, N, T]
        return x_full.unsqueeze(1).permute(0, 1, 3, 2)

    def _timeseries_path(
        self,
        V_miss: torch.Tensor,
        x_obs: torch.Tensor,
        obs_idx: torch.Tensor,
        miss_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ablation: iFFT reconstruction path (VIDA-style 2-Stage).

        V_miss [B, F, M] → iFFT → x_miss_hat [B, T, M]
        x_obs  [B, T, K] + x_miss_hat → reassemble → [B, in_dim, N, T]
        Observed positions replaced with original values (VIDA protocol).
        """
        B = x_obs.shape[0]

        # iFFT: frequency → time domain
        x_miss_hat = torch.fft.irfft(V_miss, n=self.T, dim=1, norm='ortho')  # [B, T, M]

        # Reassemble full time series
        x_full = torch.zeros(B, self.T, self.N, device=x_obs.device)
        x_full[:, :, obs_idx.cpu()] = x_obs                # observed: original values
        x_full[:, :, miss_idx.cpu()] = x_miss_hat           # missing: iFFT reconstruction

        # [B, T, N] → [B, in_dim=1, N, T]
        out = x_full.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, N, T]
        return out
