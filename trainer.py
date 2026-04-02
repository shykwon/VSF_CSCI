# CVFA Trainer
# Aligned with VIDA's DATrainer_v2 structure for fair comparison.

import torch
import torch.optim as optim
import numpy as np

from utils.metrics import masked_mae, masked_rmse


class CVFATrainer:
    """
    Training engine for CVFA model.

    Stages (aligned with VIDA for fair comparison):
        Stage 1: Forecaster pre-training (full variables, no CVFA)
        Stage 1.5: S-matrix learning (spectral alignment only)
        Stage 2: CVFA module training (forecaster frozen)
    """

    def __init__(self, args, cvfa_model, forecaster, scaler, device):
        self.args = args
        self.cvfa = cvfa_model
        self.forecaster = forecaster
        self.scaler = scaler
        self.device = device
        self.clip = args.clip

        # Loss (2-term: forecast + spectral)
        from models.loss import CVFALoss
        self.criterion = CVFALoss(
            alpha=args.alpha_loss,
            beta=args.beta_loss,
        )

        # Optimizers (created per stage)
        self.forecaster_optimizer = optim.Adam(
            self.forecaster.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.cvfa_optimizer = optim.Adam(
            self.cvfa.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Curriculum learning state
        self.step = args.step_size1
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = args.seq_out_len
        self.cl = args.cl

    # ─── Stage 1: Forecaster Training (VIDA-compatible) ───

    def train_forecaster(self, args, input, real_val, idx=None):
        """Train forecaster with full variables (VIDA protocol)."""
        self.forecaster.train()
        self.forecaster_optimizer.zero_grad()

        output = self.forecaster(input, idx=idx, args=args)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
            self.task_level += 1
        if self.cl:
            loss, _ = masked_mae(predict[:, :, :, :self.task_level],
                                 real[:, :, :, :self.task_level], 0.0)
        else:
            loss, _ = masked_mae(predict, real, 0.0)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.forecaster.parameters(), self.clip)
        self.forecaster_optimizer.step()

        rmse = masked_rmse(predict, real, 0.0)[0].item()
        self.iter += 1
        return loss.item(), rmse

    def eval_forecaster(self, args, input, real_val, idx=None):
        """Evaluate forecaster (VIDA protocol)."""
        self.forecaster.eval()
        with torch.no_grad():
            output = self.forecaster(input, idx=idx, args=args)
            output = output.transpose(1, 3)
            real = torch.unsqueeze(real_val, dim=1)
            predict = self.scaler.inverse_transform(output)
            loss, _ = masked_mae(predict, real, 0.0)
            rmse = masked_rmse(predict, real, 0.0)[0].item()
        return loss.item(), rmse

    # ─── Stage 1.5: S-only Training (L_spectral only) ───

    def train_S_only(self, args, input):
        """
        Train only the CrossSpectralEstimator (S matrix) using L_spectral.
        No forecaster forward pass needed — only FFT → Wiener → spectral loss.
        """
        # Freeze everything except cs_estimator
        for p in self.cvfa.parameters():
            p.requires_grad = False
        for p in self.cvfa.cs_estimator.parameters():
            p.requires_grad = True

        self.cvfa.cs_estimator.train()
        self.s_optimizer.zero_grad()

        B, _, N, T = input.shape
        x_signal = input[:, 0, :, :]  # [B, N, T]

        # Random masking: select ~50% as observed
        n_obs = max(N // 2, 1)
        perm = torch.randperm(N, device=input.device)
        obs_idx = perm[:n_obs].sort().values
        miss_idx = perm[n_obs:].sort().values

        # FFT on all variables
        V_all = torch.fft.rfft(x_signal, dim=-1, norm='ortho').permute(0, 2, 1)  # [B, F, N]
        V_obs = V_all[:, :, obs_idx]   # [B, F, K]
        V_miss_true = V_all[:, :, miss_idx]  # [B, F, M]

        # Wiener filter estimation
        V_miss_hat, _sigma = self.cvfa.cs_estimator(V_obs, obs_idx, miss_idx)

        # Spectral alignment loss only
        loss = self.criterion.spectral_alignment_loss(V_miss_hat, V_miss_true)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.cvfa.cs_estimator.parameters(), self.clip)
        self.s_optimizer.step()

        return loss.item()

    def eval_S_only(self, args, input):
        """Evaluate S-only spectral loss (no grad)."""
        self.cvfa.cs_estimator.eval()

        with torch.no_grad():
            B, _, N, T = input.shape
            x_signal = input[:, 0, :, :]

            n_obs = max(N // 2, 1)
            perm = torch.randperm(N, device=input.device)
            obs_idx = perm[:n_obs].sort().values
            miss_idx = perm[n_obs:].sort().values

            V_all = torch.fft.rfft(x_signal, dim=-1, norm='ortho').permute(0, 2, 1)
            V_obs = V_all[:, :, obs_idx]
            V_miss_true = V_all[:, :, miss_idx]

            V_miss_hat, _sigma = self.cvfa.cs_estimator(V_obs, obs_idx, miss_idx)
            loss = self.criterion.spectral_alignment_loss(V_miss_hat, V_miss_true)

        return loss.item()

    # ─── Stage 2: CVFA Training (Forecaster Fully Frozen) ───

    def train_cvfa(self, args, input, real_val, obs_idx, miss_idx, input_unmasked=None):
        """
        Train CVFA modules with forecaster fully frozen.
        No backbone modification — CVFA outputs [B, in_dim, N, T] directly.
        """
        # Freeze forecaster entirely (no input layer replacement)
        for p in self.forecaster.parameters():
            p.requires_grad = False
        self.forecaster.eval()  # dropout/batchnorm 비활성화

        self.cvfa.train()
        self.cvfa_optimizer.zero_grad()

        # Forward through CVFA → iFFT → forecaster-ready input
        fc_input, V_miss_hat = self.cvfa(
            input, obs_idx, miss_idx, x_full_unmasked=input_unmasked
        )
        # fc_input: [B, in_dim, N, T] — same channel dim as original backbone

        # Forecaster prediction
        output = self.forecaster(fc_input, idx=torch.arange(args.num_nodes).to(self.device), args=args)
        output = output.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)

        real = torch.unsqueeze(real_val, dim=1)

        # Ground truth spectra for spectral loss (from UNMASKED input)
        gt_source = input_unmasked if input_unmasked is not None else input
        M = miss_idx.shape[0]
        if M > 0:
            V_miss_true = torch.fft.rfft(
                gt_source[:, 0, miss_idx, :].transpose(1, 2), dim=1, norm='ortho'
            )  # [B, F, M]
        else:
            V_miss_true = None

        # Compute combined loss (2-term)
        total_loss, loss_dict = self.criterion(
            predict, real, V_miss_hat, V_miss_true
        )

        total_loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.cvfa.parameters(), self.clip)
        self.cvfa_optimizer.step()

        rmse = masked_rmse(predict, real, 0.0)[0].item()
        return total_loss.item(), rmse, loss_dict

    def eval_cvfa(self, args, input, real_val, obs_idx, miss_idx, input_unmasked=None):
        """Evaluate CVFA + Forecaster."""
        self.cvfa.eval()
        self.forecaster.eval()

        with torch.no_grad():
            fc_input, V_miss_hat = self.cvfa(
                input, obs_idx, miss_idx, x_full_unmasked=input_unmasked
            )

            output = self.forecaster(fc_input, idx=torch.arange(args.num_nodes).to(self.device), args=args)
            output = output.transpose(1, 3)
            predict = self.scaler.inverse_transform(output)

            real = torch.unsqueeze(real_val, dim=1)

            gt_source = input_unmasked if input_unmasked is not None else input
            M = miss_idx.shape[0]
            if M > 0:
                V_miss_true = torch.fft.rfft(
                    gt_source[:, 0, miss_idx, :].transpose(1, 2), dim=1, norm='ortho'
                )
            else:
                V_miss_true = None

            total_loss, loss_dict = self.criterion(
                predict, real, V_miss_hat, V_miss_true
            )
            rmse = masked_rmse(predict, real, 0.0)[0].item()

        return total_loss.item(), rmse, loss_dict
