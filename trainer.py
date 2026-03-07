# CSCI Trainer
# Aligned with VIDA's DATrainer_v2 structure for fair comparison.

import torch
import torch.optim as optim
import numpy as np

from utils.metrics import masked_mae, masked_rmse


class CSCITrainer:
    """
    Training engine for CSCI model.

    Stages (aligned with VIDA for fair comparison):
        Stage 1: Forecaster pre-training (full variables, no CSCI)
        Stage 2: CSCI module training (forecaster frozen)
        Stage 3: Joint fine-tuning (optional)
    """

    def __init__(self, args, csci_model, forecaster, scaler, device):
        self.args = args
        self.csci = csci_model
        self.forecaster = forecaster
        self.scaler = scaler
        self.device = device
        self.clip = args.clip

        # Loss
        from models.loss import CSCILoss
        self.criterion = CSCILoss(
            alpha=args.alpha_loss,
            beta=args.beta_loss,
            gamma=args.gamma_loss,
        )

        # Optimizers (created per stage)
        self.forecaster_optimizer = optim.Adam(
            self.forecaster.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.csci_optimizer = optim.Adam(
            self.csci.parameters(),
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

    # ─── Stage 2: CSCI Training (Forecaster Frozen) ───

    def train_csci(self, args, input, real_val, obs_idx, miss_idx, input_unmasked=None):
        """
        Train CSCI modules with forecaster frozen.

        Args:
            input: [B, in_dim, N, T] masked input (unobserved zeroed)
            real_val: [B, N, T] ground truth future values
            obs_idx: [K] observed variable indices
            miss_idx: [M] missing variable indices
            input_unmasked: [B, in_dim, N, T] original unmasked input (for spectral GT & tod bypass)
        """
        # Freeze forecaster (except input layers — start_conv/skip0 are trainable)
        for name, p in self.forecaster.named_parameters():
            if 'start_conv' in name or 'skip0' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.csci.train()
        self.forecaster.start_conv.train()
        self.forecaster.skip0.train()
        self.csci_optimizer.zero_grad()

        # Forward through CSCI → ForecastHead → forecaster-ready input
        fc_input, attn_bias, V_miss_hat, sigma = self.csci(
            input, obs_idx, miss_idx, x_full_unmasked=input_unmasked
        )
        # fc_input: [B, csci_in_dim, N, T] — d_model(+extra) channels

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

        # Compute combined loss
        total_loss, loss_dict = self.criterion(
            predict, real, V_miss_hat, V_miss_true, sigma
        )

        total_loss.backward()
        if self.clip is not None:
            # Clip both CSCI and trainable forecaster input layers
            trainable_params = list(self.csci.parameters()) + \
                list(self.forecaster.start_conv.parameters()) + \
                list(self.forecaster.skip0.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, self.clip)
        self.csci_optimizer.step()

        rmse = masked_rmse(predict, real, 0.0)[0].item()
        self.iter += 1
        return total_loss.item(), rmse, loss_dict

    def eval_csci(self, args, input, real_val, obs_idx, miss_idx, input_unmasked=None):
        """Evaluate CSCI + Forecaster."""
        self.csci.eval()
        self.forecaster.eval()

        with torch.no_grad():
            fc_input, attn_bias, V_miss_hat, sigma = self.csci(
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
                predict, real, V_miss_hat, V_miss_true, sigma
            )
            rmse = masked_rmse(predict, real, 0.0)[0].item()

        return total_loss.item(), rmse, loss_dict
