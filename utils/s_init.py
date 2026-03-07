# Data-driven S matrix initialization for CrossSpectralEstimator (Low-rank)
# Computes empirical cross-spectral density, then eigendecomp → U, log_diag

import torch
import numpy as np


def init_S_from_data(csci_model, dataloader, device, n_batches=50):
    """
    Initialize low-rank S parameters from empirical cross-spectral density.

    Steps:
        1. Compute empirical S from training data FFT
        2. Eigendecompose S (Hermitian → eigh)
        3. Top-r eigenvectors → U_real, U_imag
        4. Residual eigenvalues → log_diag

    Args:
        csci_model: CSCI model (accesses cs_estimator)
        dataloader: training data loader dict
        device: torch device
        n_batches: number of batches for estimation
    """
    estimator = csci_model.cs_estimator
    F = estimator.F
    N = estimator.N
    r = estimator.rank

    # Step 1: Compute empirical cross-spectral density
    S_accum = torch.zeros(F, N, N, dtype=torch.cfloat, device=device)
    count = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader['train_loader'].get_iterator()):
            if i >= n_batches:
                break
            x = torch.Tensor(x).to(device).transpose(1, 3)  # [B, in_dim, N, T]

            # FFT on signal channel
            V = torch.fft.rfft(x[:, 0, :, :], dim=-1, norm='ortho')  # [B, N, F]
            V = V.permute(0, 2, 1)  # [B, F, N]

            # Batch-averaged cross-spectral density
            S_batch = torch.einsum('bfi,bfj->fij', V, V.conj()) / V.shape[0]
            S_accum += S_batch
            count += 1

    S_empirical = S_accum / count
    # Enforce Hermitian symmetry
    S_empirical = (S_empirical + S_empirical.conj().transpose(-1, -2)) / 2

    # Step 2: Eigendecompose (eigh for Hermitian, ascending order)
    eigenvalues, eigenvectors = torch.linalg.eigh(S_empirical)  # [F,N], [F,N,N]

    # Step 3: Top-r eigenvectors (last r, since eigh returns ascending)
    top_eigenvalues = eigenvalues[:, -r:]  # [F, r]
    top_eigenvectors = eigenvectors[:, :, -r:]  # [F, N, r] complex

    # Scale eigenvectors by sqrt(eigenvalue) so U·U^H ≈ top-r component of S
    sqrt_eig = torch.sqrt(top_eigenvalues.clamp(min=1e-8)).unsqueeze(1)  # [F, 1, r]
    U_init = top_eigenvectors * sqrt_eig  # [F, N, r]

    # Step 4: Residual eigenvalues → diagonal
    # Average of remaining (non-top-r) eigenvalues as residual variance
    remaining_eigenvalues = eigenvalues[:, :-r]  # [F, N-r]
    if remaining_eigenvalues.shape[1] > 0:
        residual_var = remaining_eigenvalues.clamp(min=1e-8).mean(dim=-1)  # [F]
        # Expand to [F, N] — same residual for all variables
        diag_init = residual_var.unsqueeze(-1).expand(-1, N)  # [F, N]
    else:
        diag_init = torch.ones(F, N, device=device) * 1e-4

    # inverse softplus to get log_diag: softplus(x) = y → x = log(exp(y) - 1)
    log_diag_init = torch.log(torch.exp(diag_init) - 1.0 + 1e-8)

    # Write to model parameters
    estimator.U_real.data.copy_(U_init.real)
    estimator.U_imag.data.copy_(U_init.imag)
    estimator.log_diag.data.copy_(log_diag_init)

    # Verify reconstruction quality
    S_reconstructed = estimator.get_S()
    recon_error = (S_reconstructed - S_empirical).abs().mean().item()
    explained = 1.0 - recon_error / (S_empirical.abs().mean().item() + 1e-8)

    print(f"  S initialized (low-rank r={r}): mean|S|={S_empirical.abs().mean():.4f}, "
          f"recon_error={recon_error:.4f}, explained={explained:.2%}, batches={count}")
