# Module 1: Spectral Encoder
# Reference: docs/project_develop_plan.md Section 2

import torch
import torch.nn as nn


class SpectralEncoder(nn.Module):
    """
    Time domain → Frequency domain via real FFT.

    Input:  x [B, T, K] (real)
    Output: V [B, F, K] (complex), F = T // 2 + 1
    """

    def __init__(self, T: int, norm: str = 'ortho'):
        super().__init__()
        self.T = T
        self.norm = norm
        self.F = T // 2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, K] real-valued time series
        Returns:
            V: [B, F, K] complex-valued frequency representation
        """
        # dim=1: FFT along time axis
        V = torch.fft.rfft(x, dim=1, norm=self.norm)  # [B, F, K]
        return V
