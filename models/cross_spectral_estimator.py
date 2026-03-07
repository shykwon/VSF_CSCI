# Module 2: Cross-Spectral Estimator (Low-rank S)
# Reference: docs/project_develop_plan.md Section 3

import torch
import torch.nn as nn
import torch.nn.functional as nnF


class CrossSpectralEstimator(nn.Module):
    """
    Estimate missing variable frequency representations via Wiener filter
    using a learnable low-rank cross-spectral density matrix S.

    S(f) = U(f) · U(f)^H + diag(d(f))
      - U: [F, N, r] — r共通 spectral patterns (complex)
      - d: [F, N] — per-variable residual variance (positive via softplus)
      - Hermitian PSD guaranteed by construction

    Key equations:
        V_miss(f) = S_mo(f) · [S_oo(f) + λI]^(-1) · V_obs(f)
        σ_miss(f) = diag(S_mm) - diag(S_mo · S_oo^(-1) · S_om)
    """

    def __init__(self, N: int, F: int, lambda_reg: float = 0.1, rank: int = 16):
        super().__init__()
        self.N = N
        self.F = F
        self.lambda_reg = lambda_reg
        self.rank = rank

        # Low-rank factors: S = U · U^H + diag(softplus(log_diag))
        self.U_real = nn.Parameter(torch.randn(F, N, rank) * 0.01)
        self.U_imag = nn.Parameter(torch.zeros(F, N, rank))
        self.log_diag = nn.Parameter(torch.zeros(F, N))

    def get_S(self) -> torch.Tensor:
        """
        Construct Hermitian PSD cross-spectral matrix from low-rank factors.
        S(f) = U(f) · U(f)^H + diag(d(f))
        """
        U = torch.complex(self.U_real, self.U_imag)  # [F, N, r]
        S_lr = torch.bmm(U, U.conj().transpose(-1, -2))  # [F, N, N] Hermitian
        d = nnF.softplus(self.log_diag)  # [F, N] positive
        S_diag = torch.diag_embed(d.to(S_lr.dtype))  # [F, N, N]
        return S_lr + S_diag

    def forward(
        self,
        V_obs: torch.Tensor,
        obs_idx: torch.Tensor,
        miss_idx: torch.Tensor,
    ) -> tuple:
        """
        Args:
            V_obs: [B, F, K] complex
            obs_idx: [K] int tensor — observed indices
            miss_idx: [M] int tensor — missing indices
        Returns:
            V_miss_hat: [B, F, M] complex
            sigma: [B, F, M] real (non-negative)
        """
        B, F, K = V_obs.shape
        M = miss_idx.shape[0]
        S = self.get_S()  # [F, N, N]

        # Extract sub-blocks
        S_oo = S[:, obs_idx][:, :, obs_idx]    # [F, K, K]
        S_mo = S[:, miss_idx][:, :, obs_idx]   # [F, M, K]
        S_mm = S[:, miss_idx][:, :, miss_idx]  # [F, M, M]

        # Regularized inverse: [S_oo + λI]^(-1)
        reg = self.lambda_reg * torch.eye(K, device=V_obs.device, dtype=S_oo.dtype).unsqueeze(0)
        S_oo_inv = torch.linalg.inv(S_oo + reg)  # [F, K, K]

        # Wiener filter: W = S_mo · S_oo_inv  [F, M, K]
        W = torch.bmm(S_mo, S_oo_inv)  # [F, M, K]

        # Estimate missing spectra: V_miss = W · V_obs
        W_expand = W.unsqueeze(0).expand(B, -1, -1, -1)  # [B, F, M, K]
        V_obs_expand = V_obs.unsqueeze(-1)  # [B, F, K, 1]
        V_miss_hat = torch.matmul(
            W_expand.reshape(B * F, M, K),
            V_obs_expand.reshape(B * F, K, 1)
        ).reshape(B, F, M)  # [B, F, M]

        # Clamp V_miss_hat amplitude to prevent explosion (preserve phase)
        obs_scale = V_obs.abs().max().clamp(min=1.0)
        max_amp = obs_scale * 10
        scale = V_miss_hat.abs() / (max_amp + 1e-8)
        V_miss_hat = V_miss_hat / scale.clamp(min=1.0)

        # Uncertainty: σ = diag(S_mm) - diag(W · S_om)
        S_om = S_mo.conj().transpose(-1, -2)  # [F, K, M]
        WS_om = torch.bmm(W, S_om)  # [F, M, M]
        diag_S_mm = torch.diagonal(S_mm, dim1=-2, dim2=-1)    # [F, M]
        diag_WS_om = torch.diagonal(WS_om, dim1=-2, dim2=-1)  # [F, M]
        sigma = (diag_S_mm - diag_WS_om).real  # [F, M]
        sigma = sigma.unsqueeze(0).expand(B, -1, -1)  # [B, F, M]
        sigma = torch.clamp(sigma, min=0)

        return V_miss_hat, sigma

    def compute_coherence(self, S, obs_idx, miss_idx):
        """
        Compute spectral coherence between observed and missing variables.
        Returns: coherence: [F, M] real in [0, 1]
        """
        S_oo = S[:, obs_idx][:, :, obs_idx]
        S_mo = S[:, miss_idx][:, :, obs_idx]
        S_mm = S[:, miss_idx][:, :, miss_idx]

        cross_power = (S_mo.abs() ** 2).sum(dim=-1)  # [F, M]
        diag_mm = torch.diagonal(S_mm, dim1=-2, dim2=-1).real  # [F, M]
        diag_oo_sum = torch.diagonal(S_oo, dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True)

        denom = diag_mm * diag_oo_sum + 1e-8
        coherence = cross_power / denom
        coherence = torch.clamp(coherence, 0, 1)
        return coherence

    def wiener_filter(self, V_obs, obs_idx, miss_idx):
        """Single-round Wiener filter (reusable for hierarchical propagation)."""
        B, F, K = V_obs.shape
        S = self.get_S()

        obs_idx_t = torch.as_tensor(obs_idx, device=V_obs.device)
        miss_idx_t = torch.as_tensor(miss_idx, device=V_obs.device)

        S_oo = S[:, obs_idx_t][:, :, obs_idx_t]
        S_mo = S[:, miss_idx_t][:, :, obs_idx_t]
        S_mm = S[:, miss_idx_t][:, :, miss_idx_t]

        reg = self.lambda_reg * torch.eye(K, device=V_obs.device, dtype=S_oo.dtype).unsqueeze(0)
        S_oo_inv = torch.linalg.inv(S_oo + reg)
        W = torch.bmm(S_mo, S_oo_inv)

        W_expand = W.unsqueeze(0).expand(B, -1, -1, -1)
        V_obs_expand = V_obs.unsqueeze(-1)
        M = len(miss_idx)
        V_miss_hat = torch.matmul(
            W_expand.reshape(B * F, M, K),
            V_obs_expand.reshape(B * F, K, 1)
        ).reshape(B, F, M)

        # Clamp V_miss_hat amplitude to prevent explosion (preserve phase)
        obs_scale = V_obs.abs().max().clamp(min=1.0)
        max_amp = obs_scale * 10
        scale = V_miss_hat.abs() / (max_amp + 1e-8)
        V_miss_hat = V_miss_hat / scale.clamp(min=1.0)

        S_om = S_mo.conj().transpose(-1, -2)
        WS_om = torch.bmm(W, S_om)
        diag_S_mm = torch.diagonal(S_mm, dim1=-2, dim2=-1)
        diag_WS_om = torch.diagonal(WS_om, dim1=-2, dim2=-1)
        sigma = (diag_S_mm - diag_WS_om).real
        sigma = sigma.unsqueeze(0).expand(B, -1, -1)
        sigma = torch.clamp(sigma, min=0)

        return V_miss_hat, sigma

    def forward_hierarchical(
        self,
        V_obs: torch.Tensor,
        obs_idx: torch.Tensor,
        miss_idx: torch.Tensor,
        n_rounds: int = 2,
        base_threshold: float = 0.3,
        threshold_decay: float = 0.1,
    ) -> tuple:
        """Hierarchical propagation: multi-round Wiener filter."""
        B, F, _ = V_obs.shape
        S = self.get_S()
        device = V_obs.device

        current_obs = V_obs.clone()
        current_obs_idx = obs_idx.cpu().tolist()
        remaining_miss = miss_idx.cpu().tolist()

        miss_idx_list = miss_idx.cpu().tolist()
        M = len(miss_idx_list)
        all_V_miss = torch.zeros(B, F, M, dtype=V_obs.dtype, device=device)
        all_sigma = torch.zeros(B, F, M, device=device)

        for round_i in range(n_rounds):
            if not remaining_miss:
                break

            obs_t = torch.tensor(current_obs_idx, device=device)
            rem_t = torch.tensor(remaining_miss, device=device)
            coherence = self.compute_coherence(S, obs_t, rem_t)

            threshold = base_threshold - round_i * threshold_decay
            mean_coh = coherence.mean(dim=0)
            high_coh_mask = mean_coh > threshold

            if round_i == n_rounds - 1:
                high_coh_mask = torch.ones_like(high_coh_mask, dtype=torch.bool)

            if not high_coh_mask.any():
                continue

            easy_local = high_coh_mask.nonzero(as_tuple=True)[0].cpu().tolist()
            easy_miss = [remaining_miss[i] for i in easy_local]

            V_easy, sigma_easy = self.wiener_filter(current_obs, current_obs_idx, easy_miss)

            for local_i, global_i in enumerate(easy_miss):
                orig_pos = miss_idx_list.index(global_i)
                all_V_miss[:, :, orig_pos] = V_easy[:, :, local_i]
                all_sigma[:, :, orig_pos] = sigma_easy[:, :, local_i]

            current_obs = torch.cat([current_obs, V_easy], dim=-1)
            current_obs_idx.extend(easy_miss)
            remaining_miss = [i for i in remaining_miss if i not in easy_miss]

        return all_V_miss, all_sigma
