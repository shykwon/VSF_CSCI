# Data-driven S matrix initialization for CrossSpectralEstimator
# Computes empirical cross-spectral density from training data

import torch


def init_S_from_data(csci_model, dataloader, device, n_batches=50):
    """
    Initialize S matrix from empirical cross-spectral density.

    Instead of starting from Identity (all variables independent),
    use actual data correlations so Wiener filter produces
    meaningful estimates from the start.

    Args:
        csci_model: CSCI model (accesses cs_estimator.S_real/S_imag)
        dataloader: training data loader (dict with 'train_loader')
        device: torch device
        n_batches: number of batches to use for estimation
    """
    F = csci_model.cs_estimator.F
    N = csci_model.cs_estimator.N
    S_accum = torch.zeros(F, N, N, dtype=torch.cfloat, device=device)
    count = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader['train_loader'].get_iterator()):
            if i >= n_batches:
                break
            x = torch.Tensor(x).to(device).transpose(1, 3)  # [B, in_dim, N, T]

            # FFT on signal channel only
            V = torch.fft.rfft(x[:, 0, :, :], dim=-1, norm='ortho')  # [B, N, F]
            V = V.permute(0, 2, 1)  # [B, F, N]

            # Batch-averaged cross-spectral density: S_ij = E[V_i * conj(V_j)]
            S_batch = torch.einsum('bfi,bfj->fij', V, V.conj()) / V.shape[0]
            S_accum += S_batch
            count += 1

    S_init = S_accum / count

    # Enforce Hermitian symmetry
    S_init = (S_init + S_init.conj().transpose(-1, -2)) / 2

    # Write to model parameters
    csci_model.cs_estimator.S_real.data.copy_(S_init.real)
    csci_model.cs_estimator.S_imag.data.copy_(S_init.imag)

    print(f"  S initialized from data: mean|S|={S_init.abs().mean():.4f}, "
          f"max|S|={S_init.abs().max():.4f}, batches={count}")
