# Module 2: Cross-Spectral Estimator
# Reference: docs/project_develop_plan.md Section 3

import torch
import torch.nn as nn


class CrossSpectralEstimator(nn.Module):
    """
    Estimate missing variable frequency representations via Wiener filter
    using a learnable cross-spectral density matrix S.

    Key equations:
        V_miss(f) = S_mo(f) · [S_oo(f) + λI]^(-1) · V_obs(f)
        σ_miss(f) = diag(S_mm) - diag(S_mo · S_oo^(-1) · S_om)

    Input:
        V_obs: [B, F, K] complex — observed variable spectra
        obs_idx: list[int] — observed variable indices
        miss_idx: list[int] — missing variable indices
    Output:
        V_miss_hat: [B, F, M] complex — estimated missing spectra
        sigma: [B, F, M] real — estimation uncertainty
    """

    def __init__(self, N: int, F: int, lambda_reg: float = 1e-3):
        super().__init__()
        self.N = N
        self.F = F
        self.lambda_reg = lambda_reg

        # Learnable cross-spectral matrix (real + imaginary parts)
        # Initialized: S_real = I (identity), S_imag = 0
        self.S_real = nn.Parameter(torch.eye(N).unsqueeze(0).repeat(F, 1, 1))  # [F, N, N]
        self.S_imag = nn.Parameter(torch.zeros(F, N, N))  # [F, N, N]

    def get_S(self) -> torch.Tensor:
        """
        Construct Hermitian-symmetric cross-spectral matrix.
        Hermitian: S_ij = conj(S_ji)
        """
        S_real_sym = (self.S_real + self.S_real.transpose(-1, -2)) / 2
        S_imag_antisym = (self.S_imag - self.S_imag.transpose(-1, -2)) / 2
        S = torch.complex(S_real_sym, S_imag_antisym)  # [F, N, N]
        return S

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
        # V_obs [B, F, K] → need batch-freq matmul
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

        Coherence(f) = |S_mo(f)|^2 / (diag(S_mm(f)) * diag(S_oo(f)))
        Averaged over observed variables for each missing variable.

        Returns:
            coherence: [F, M] real in [0, 1]
        """
        S_oo = S[:, obs_idx][:, :, obs_idx]    # [F, K, K]
        S_mo = S[:, miss_idx][:, :, obs_idx]   # [F, M, K]
        S_mm = S[:, miss_idx][:, :, miss_idx]  # [F, M, M]

        # |S_mo|^2 summed over observed vars → [F, M]
        cross_power = (S_mo.abs() ** 2).sum(dim=-1)  # [F, M]

        # Normalization: diag(S_mm) * sum(diag(S_oo))
        diag_mm = torch.diagonal(S_mm, dim1=-2, dim2=-1).real  # [F, M]
        diag_oo_sum = torch.diagonal(S_oo, dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True)  # [F, 1]

        denom = diag_mm * diag_oo_sum + 1e-8
        coherence = cross_power / denom  # [F, M]
        coherence = torch.clamp(coherence, 0, 1)
        return coherence

    def wiener_filter(self, V_obs, obs_idx, miss_idx):
        """
        Single-round Wiener filter (reusable for hierarchical propagation).

        Args:
            V_obs: [B, F, K] complex
            obs_idx: list/tensor of observed indices
            miss_idx: list/tensor of missing indices
        Returns:
            V_miss_hat: [B, F, M] complex
            sigma: [B, F, M] real
        """
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
        """
        Hierarchical propagation: multi-round Wiener filter.

        Round 1: Observed → high-coherence missing variables
        Round 2: Observed + Round1 restored → remaining missing variables
        ...

        Args:
            V_obs: [B, F, K] complex
            obs_idx: [K] int tensor
            miss_idx: [M] int tensor
            n_rounds: number of propagation rounds
            base_threshold: initial coherence threshold
            threshold_decay: threshold reduction per round
        Returns:
            V_miss_hat: [B, F, M] complex — all missing variables estimated
            sigma: [B, F, M] real — uncertainty for all missing variables
        """
        B, F, _ = V_obs.shape
        S = self.get_S()
        device = V_obs.device

        current_obs = V_obs.clone()
        current_obs_idx = obs_idx.cpu().tolist()
        remaining_miss = miss_idx.cpu().tolist()

        # Track results for all missing variables (in original miss_idx order)
        miss_idx_list = miss_idx.cpu().tolist()
        M = len(miss_idx_list)
        all_V_miss = torch.zeros(B, F, M, dtype=V_obs.dtype, device=device)
        all_sigma = torch.zeros(B, F, M, device=device)

        for round_i in range(n_rounds):
            if not remaining_miss:
                break

            # Compute coherence between current observed and remaining missing
            obs_t = torch.tensor(current_obs_idx, device=device)
            rem_t = torch.tensor(remaining_miss, device=device)
            coherence = self.compute_coherence(S, obs_t, rem_t)  # [F, len(remaining)]

            # Select high-coherence variables
            threshold = base_threshold - round_i * threshold_decay
            mean_coh = coherence.mean(dim=0)  # [len(remaining)]
            high_coh_mask = mean_coh > threshold

            if round_i == n_rounds - 1:
                # Last round: restore all remaining regardless of threshold
                high_coh_mask = torch.ones_like(high_coh_mask, dtype=torch.bool)

            if not high_coh_mask.any():
                continue

            # Get indices of easy-to-restore variables
            easy_local = high_coh_mask.nonzero(as_tuple=True)[0].cpu().tolist()
            easy_miss = [remaining_miss[i] for i in easy_local]

            # Wiener filter for easy variables
            V_easy, sigma_easy = self.wiener_filter(current_obs, current_obs_idx, easy_miss)

            # Store results in correct positions
            for local_i, global_i in enumerate(easy_miss):
                orig_pos = miss_idx_list.index(global_i)
                all_V_miss[:, :, orig_pos] = V_easy[:, :, local_i]
                all_sigma[:, :, orig_pos] = sigma_easy[:, :, local_i]

            # Add restored variables to observation pool
            current_obs = torch.cat([current_obs, V_easy], dim=-1)
            current_obs_idx.extend(easy_miss)
            remaining_miss = [i for i in remaining_miss if i not in easy_miss]

        return all_V_miss, all_sigma
