# Forecast Head Adapter
# Bridges CSCI output (embedding dim) to forecaster input (time series dim).
# Also supports time-series path for ablation (iFFT reconstruction, like VIDA).

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """
    Adapts CSCI output to forecaster input format.

    Two modes:
      1. 'embedding' (default): h [B, T, N, d_model] → permute → [B, d_model, N, T]
         - CSCI의 핵심 경로. d_model 차원 임베딩을 그대로 백본에 주입.
         - 백본의 start_conv/skip0만 교체하여 d_model 채널 수용.

      2. 'timeseries' (ablation): V_miss → iFFT → 시계열 복원 → [B, 1, N, T]
         - VIDA와 동일한 2-Stage 방식. H2 가설 검증용.
         - 관측값은 원본으로 대체 (VIDA 프로토콜).

    Usage:
        head = ForecastHead(mode='embedding', d_model=64)
        fc_input = head(h=h, ...)               # embedding mode → [B, d_model, N, T]
        fc_input = head(V_miss=V, x_obs=x, ...)  # timeseries mode → [B, 1, N, T]
    """

    def __init__(self, mode: str, N: int, T: int, d_model: int = 64, in_dim: int = 1):
        """
        Args:
            mode: 'embedding' or 'timeseries'
            N: total number of variables
            T: input sequence length
            d_model: CSCI embedding dimension (only for embedding mode)
            in_dim: forecaster input channels (only for timeseries mode)
        """
        super().__init__()
        self.mode = mode
        self.N = N
        self.T = T
        self.d_model = d_model
        self.in_dim = in_dim

        if mode == 'embedding':
            # No projection — pass d_model dims directly to backbone
            pass
        elif mode == 'timeseries':
            # No learnable params needed — just iFFT + reassemble
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'embedding' or 'timeseries'.")

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
        else:
            return self._timeseries_path(V_miss, x_obs, obs_idx, miss_idx)

    def _embedding_path(self, h: torch.Tensor) -> torch.Tensor:
        """
        h [B, T, N, d_model] → permute → [B, d_model, N, T]
        No projection — full embedding passed to backbone.
        """
        return h.permute(0, 3, 2, 1)  # [B, d_model, N, T]

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
