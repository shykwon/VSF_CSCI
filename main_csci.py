#!/usr/bin/env python3
# CSCI Main Training Script
# Structure aligned with VIDA's main_vida.py for fair comparison.

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import os
import random
import datetime

from utils.data_utils import load_dataset
from utils.graph_utils import load_adj
from utils.masking import (
    zero_out_remaining_input,
    get_node_random_idx_split,
    get_idx_subset_from_idx_all_nodes,
    get_curriculum_mask_ratio,
)
from utils.metrics import metric
from utils.eval_metrics import evaluate_spectral_metrics
from utils.result_tracker import ResultTracker

from models.csci import CSCI
from forecasters.net import gtnet
from trainer import CSCITrainer
from utils.s_init import init_S_from_data


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def build_args():
    parser = argparse.ArgumentParser(description='CSCI: Cross-Spectral Coherence Imputation')

    # ── Device & Data ──
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data', type=str, default='data/METR-LA')
    parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_mx.pkl')
    parser.add_argument('--seed', type=int, default=3407)

    # ── Model: MTGNN Forecaster (VIDA-compatible) ──
    parser.add_argument('--model_name', type=str, default='mtgnn')
    parser.add_argument('--gcn_true', type=str_to_bool, default=True)
    parser.add_argument('--buildA_true', type=str_to_bool, default=True)
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--num_nodes', type=int, default=207)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--subgraph_size', type=int, default=20)
    parser.add_argument('--node_dim', type=int, default=40)
    parser.add_argument('--dilation_exponential', type=int, default=1)
    parser.add_argument('--conv_channels', type=int, default=32)
    parser.add_argument('--residual_channels', type=int, default=32)
    parser.add_argument('--skip_channels', type=int, default=64)
    parser.add_argument('--end_channels', type=int, default=128)
    parser.add_argument('--in_dim', type=int, default=1)
    parser.add_argument('--seq_in_len', type=int, default=12)
    parser.add_argument('--seq_out_len', type=int, default=12)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--propalpha', type=float, default=0.05)
    parser.add_argument('--tanhalpha', type=float, default=3)
    parser.add_argument('--num_split', type=int, default=1)
    parser.add_argument('--cl', type=str_to_bool, default=True)
    parser.add_argument('--load_static_feature', type=str_to_bool, default=False)
    parser.add_argument('--adj_identity_train_test', type=str_to_bool, default=False)

    # ── Model: CSCI ──
    parser.add_argument('--d_model', type=int, default=64, help='CSCI embedding dimension')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='Wiener filter regularization')
    parser.add_argument('--head_mode', type=str, default='embedding',
                        choices=['embedding', 'timeseries'],
                        help="'embedding': CSCI path (no iFFT), 'timeseries': ablation (iFFT, VIDA-style)")
    parser.add_argument('--s_rank', type=int, default=16, help='Low-rank S matrix rank')
    parser.add_argument('--s_epochs', type=int, default=20, help='Stage 1.5: S-only training epochs')
    parser.add_argument('--n_rounds', type=int, default=1,
                        help='Hierarchical propagation rounds (1=disabled, 2=default for 85%%)')
    parser.add_argument('--coherence_threshold', type=float, default=0.3,
                        help='Base coherence threshold for hierarchical propagation')
    parser.add_argument('--threshold_decay', type=float, default=0.1,
                        help='Threshold reduction per propagation round')
    parser.add_argument('--use_curriculum', type=str_to_bool, default=True,
                        help='Enable masking curriculum for CSCI training')

    # ── Training ──
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--fc_epochs', type=int, default=100, help='Forecaster training epochs')
    parser.add_argument('--csci_epochs', type=int, default=100, help='CSCI training epochs')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--step_size1', type=int, default=2500)
    parser.add_argument('--step_size2', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=50)

    # ── Loss Weights ──
    parser.add_argument('--alpha_loss', type=float, default=0.6, help='Forecast loss weight')
    parser.add_argument('--beta_loss', type=float, default=0.3, help='Spectral alignment loss weight')
    parser.add_argument('--gamma_loss', type=float, default=0.1, help='Uncertainty calibration loss weight')

    # ── VSF Evaluation (VIDA protocol) ──
    parser.add_argument('--runs', type=int, default=1, help='Number of model training runs')
    parser.add_argument('--random_node_idx_split_runs', type=int, default=100)
    parser.add_argument('--lower_limit_random_node_selections', type=int, default=15)
    parser.add_argument('--upper_limit_random_node_selections', type=int, default=15)
    parser.add_argument('--predefined_S', type=str_to_bool, default=False)
    parser.add_argument('--predefined_S_frac', type=int, default=15)

    # ── Experiment ──
    parser.add_argument('--expid', type=int, default=1)

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, runid):
    device = torch.device(args.device)
    dataloader = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    args.num_nodes = dataloader['train_loader'].num_nodes
    print(f"Number of variables/nodes = {args.num_nodes}")

    dataset_name = args.data.strip().split('/')[-1].strip()
    # METR-LA has time_of_day as 2nd channel (in_dim=2 in original data)
    # Stage 1 forecaster trains with in_dim from data for fair VIDA comparison.
    # CSCI spectral analysis uses channel 0 only (handled internally).

    path_model_save = f"./saved_models/{args.model_name}/{dataset_name}/seed{args.seed}/"
    os.makedirs(path_model_save, exist_ok=True)

    # ── Build adjacency matrix (VIDA-compatible) ──
    if dataset_name == "METR-LA":
        predefined_A = load_adj(args.adj_data)
        predefined_A = torch.tensor(predefined_A) - torch.eye(dataloader['total_num_nodes'])
        predefined_A = predefined_A.to(device)
    else:
        predefined_A = None

    if args.node_dim >= args.num_nodes:
        args.node_dim = args.num_nodes
        args.subgraph_size = args.num_nodes

    # ── Build Forecaster (MTGNN, VIDA-compatible) ──
    forecaster = gtnet(
        args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
        device, predefined_A=predefined_A,
        dropout=args.dropout, subgraph_size=args.subgraph_size,
        node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels, residual_channels=args.residual_channels,
        skip_channels=args.skip_channels, end_channels=args.end_channels,
        seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
        layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
        layer_norm_affline=True,
    ).to(device)
    print(f"Forecaster receptive field: {forecaster.receptive_field}")
    print(f"Forecaster params: {sum(p.numel() for p in forecaster.parameters()):,}")

    # ── Build CSCI Model ──
    csci_model = CSCI(args).to(device)
    print(f"CSCI params: {sum(p.numel() for p in csci_model.parameters()):,}")

    # ── Result Tracker ──
    exp_name = f"csci_{dataset_name}_exp{args.expid}_run{runid}_seed{args.seed}"
    tracker = ResultTracker("results", exp_name)
    tracker.set_config(args)

    # ── Trainer ──
    engine = CSCITrainer(args, csci_model, forecaster, scaler, device)

    # ══════════════════════════════════════════════════
    #  Stage 1: Forecaster Training (VIDA-compatible)
    # ══════════════════════════════════════════════════
    print("=" * 50)
    print("  Stage 1: Forecaster Training")
    print("=" * 50)

    minl = 1e5
    for epoch in range(1, args.fc_epochs + 1):
        train_loss, train_rmse = [], []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.train_forecaster(args, trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_rmse.append(metrics[1])
            if iter % args.print_every == 0:
                print(f"  Iter {iter:03d}, Loss: {metrics[0]:.4f}, RMSE: {metrics[1]:.4f}")

        # Validation
        val_loss, val_rmse = [], []
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.eval_forecaster(args, testx, testy[:, 0, :, :])
            val_loss.append(metrics[0])
            val_rmse.append(metrics[1])

        mtrain_loss = np.mean(train_loss)
        mval_loss = np.mean(val_loss)
        mval_rmse = np.mean(val_rmse)
        t2 = time.time()

        tracker.log_train_epoch(epoch, mtrain_loss, mval_loss, mval_rmse, stage="forecaster")
        print(f"  Epoch {epoch:03d} | Train: {mtrain_loss:.4f} | Val: {mval_loss:.4f} | "
              f"RMSE: {mval_rmse:.4f} | Time: {t2-t1:.1f}s")

        if mval_loss < minl:
            torch.save(forecaster.state_dict(),
                       os.path.join(path_model_save, f"forecaster_exp{args.expid}_run{runid}.pth"))
            minl = mval_loss

    # Load best forecaster
    forecaster.load_state_dict(torch.load(
        os.path.join(path_model_save, f"forecaster_exp{args.expid}_run{runid}.pth"),
        weights_only=True,
    ))
    print("  Forecaster loaded (best val loss).\n")

    # ══════════════════════════════════════════════════
    #  Stage 1.5: S-only Training (spectral loss only)
    # ══════════════════════════════════════════════════

    # Initialize S matrix from training data via eigendecomposition
    init_S_from_data(csci_model, dataloader, device, n_batches=50)

    # S-only optimizer (only cs_estimator parameters)
    engine.s_optimizer = torch.optim.Adam(
        csci_model.cs_estimator.parameters(),
        lr=args.learning_rate * 10,  # faster lr for S convergence
        weight_decay=args.weight_decay,
    )

    if args.s_epochs > 0:
        print("=" * 50)
        print("  Stage 1.5: S-only Training (L_spectral)")
        print("=" * 50)

        for epoch in range(1, args.s_epochs + 1):
            train_loss = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()

            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device).transpose(1, 3)
                loss = engine.train_S_only(args, trainx)
                train_loss.append(loss)

            # Validation
            val_loss = []
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device).transpose(1, 3)
                loss = engine.eval_S_only(args, testx)
                val_loss.append(loss)

            t2 = time.time()
            print(f"  Epoch {epoch:03d} | Train: {np.mean(train_loss):.4f} | "
                  f"Val: {np.mean(val_loss):.4f} | Time: {t2-t1:.1f}s")

        # Re-enable all CSCI params
        for p in csci_model.parameters():
            p.requires_grad = True

    # ══════════════════════════════════════════════════
    #  Stage 2: CSCI Training (Forecaster input layer replaced)
    # ══════════════════════════════════════════════════

    # Replace backbone input layers to accept d_model-dim embedding
    csci_in_dim = csci_model.get_csci_in_dim()
    print(f"  Replacing forecaster input layers: in_dim {args.in_dim} → {csci_in_dim}")

    forecaster.start_conv = nn.Conv2d(
        in_channels=csci_in_dim,
        out_channels=args.residual_channels,
        kernel_size=(1, 1),
    ).to(device)

    if args.seq_in_len > forecaster.receptive_field:
        forecaster.skip0 = nn.Conv2d(
            in_channels=csci_in_dim,
            out_channels=args.skip_channels,
            kernel_size=(1, args.seq_in_len),
            bias=True,
        ).to(device)
    else:
        forecaster.skip0 = nn.Conv2d(
            in_channels=csci_in_dim,
            out_channels=args.skip_channels,
            kernel_size=(1, forecaster.receptive_field),
            bias=True,
        ).to(device)

    print(f"  start_conv: Conv2d({csci_in_dim}, {args.residual_channels}, (1,1))")
    print(f"  skip0: Conv2d({csci_in_dim}, {args.skip_channels}, ...)")

    # Rebuild optimizer to include new input layers
    engine.csci_optimizer = torch.optim.Adam(
        list(csci_model.parameters()) +
        list(forecaster.start_conv.parameters()) +
        list(forecaster.skip0.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("=" * 50)
    print("  Stage 2: CSCI Training")
    print("=" * 50)

    minl = 1e5
    early_stop_counter = 0

    for epoch in range(1, args.csci_epochs + 1):
        train_loss, train_rmse = [], []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        # Determine observation ratio for this epoch
        # mask_ratio in get_idx_subset_from_idx_all_nodes = fraction of nodes SELECTED (observed)
        # curriculum returns missing rate, so obs_ratio = 1 - missing_rate
        if args.use_curriculum:
            miss_rate = get_curriculum_mask_ratio(
                epoch - 1, args.csci_epochs, max_ratio=0.85
            )
            obs_ratio = max(1.0 - miss_rate, 0.05)  # at least 5% observed
        else:
            obs_ratio = 0.15  # VIDA default: 15% observed

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)

            # Random subset selection for training (curriculum-aware)
            # Ensure at least 1 missing variable for CSCI training
            all_idx = torch.arange(args.num_nodes).to(device)
            idx_subset = get_idx_subset_from_idx_all_nodes(all_idx, mask_ratio=obs_ratio)
            if len(idx_subset) >= args.num_nodes:
                idx_subset = idx_subset[:args.num_nodes - 1]
            obs_idx = torch.tensor(idx_subset, dtype=torch.long).to(device)
            miss_idx = torch.tensor(
                np.setdiff1d(np.arange(args.num_nodes), idx_subset),
                dtype=torch.long
            ).to(device)

            # Zero out unobserved variables
            trainx_masked = zero_out_remaining_input(trainx.clone(), idx_subset, args.device)

            metrics = engine.train_csci(args, trainx_masked, trainy[:, 0, :, :], obs_idx, miss_idx,
                                        input_unmasked=trainx)
            train_loss.append(metrics[0])
            train_rmse.append(metrics[1])

            if iter % args.print_every == 0:
                ld = metrics[2]
                print(f"  Iter {iter:03d} | Total: {ld['total']:.4f} | "
                      f"FC: {ld['forecast']:.4f} | Spec: {ld.get('spectral', 0):.4f} | "
                      f"Unc: {ld.get('uncertainty', 0):.4f}")

        # Validation
        val_loss, val_rmse = [], []
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device).transpose(1, 3)

            all_idx = torch.arange(args.num_nodes).to(device)
            idx_subset = get_idx_subset_from_idx_all_nodes(all_idx, mask_ratio=obs_ratio)
            if len(idx_subset) >= args.num_nodes:
                idx_subset = idx_subset[:args.num_nodes - 1]
            obs_idx = torch.tensor(idx_subset, dtype=torch.long).to(device)
            miss_idx = torch.tensor(
                np.setdiff1d(np.arange(args.num_nodes), idx_subset),
                dtype=torch.long
            ).to(device)

            testx_masked = zero_out_remaining_input(testx.clone(), idx_subset, args.device)
            metrics = engine.eval_csci(args, testx_masked, testy[:, 0, :, :], obs_idx, miss_idx,
                                       input_unmasked=testx)
            val_loss.append(metrics[0])
            val_rmse.append(metrics[1])

        mtrain_loss = np.mean(train_loss)
        mval_loss = np.mean(val_loss)
        mval_rmse = np.mean(val_rmse)
        t2 = time.time()

        tracker.log_train_epoch(epoch, mtrain_loss, mval_loss, mval_rmse, stage="csci")
        miss_pct = f"{miss_rate*100:.0f}%" if args.use_curriculum else "fixed"
        print(f"  Epoch {epoch:03d} | Train: {mtrain_loss:.4f} | Val: {mval_loss:.4f} | "
              f"RMSE: {mval_rmse:.4f} | Miss: {miss_pct} | Time: {t2-t1:.1f}s | "
              f"ES: {early_stop_counter}/{args.patience}")

        if mval_loss < minl:
            torch.save(csci_model.state_dict(),
                       os.path.join(path_model_save, f"csci_exp{args.expid}_run{runid}.pth"))
            # Also save replaced input layers (start_conv, skip0)
            torch.save({
                'start_conv': forecaster.start_conv.state_dict(),
                'skip0': forecaster.skip0.state_dict(),
            }, os.path.join(path_model_save, f"fc_input_layers_exp{args.expid}_run{runid}.pth"))
            minl = mval_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print("  Early stopping.")
                break

    # Load best CSCI + input layers
    csci_model.load_state_dict(torch.load(
        os.path.join(path_model_save, f"csci_exp{args.expid}_run{runid}.pth"),
        weights_only=True,
    ))
    input_layers = torch.load(
        os.path.join(path_model_save, f"fc_input_layers_exp{args.expid}_run{runid}.pth"),
        weights_only=True,
    )
    forecaster.start_conv.load_state_dict(input_layers['start_conv'])
    forecaster.skip0.load_state_dict(input_layers['skip0'])
    print("  CSCI + input layers loaded (best val loss).\n")

    # ══════════════════════════════════════════════════
    #  Inference (VIDA protocol: 100 random splits)
    # ══════════════════════════════════════════════════
    print("=" * 50)
    print("  Inference (100 random splits)")
    print("=" * 50)

    # Build separate oracle forecaster with original Stage 1 weights
    # (current forecaster has replaced input layers for d_model channels)
    oracle_forecaster = gtnet(
        args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
        device, predefined_A=predefined_A,
        dropout=args.dropout, subgraph_size=args.subgraph_size,
        node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels, residual_channels=args.residual_channels,
        skip_channels=args.skip_channels, end_channels=args.end_channels,
        seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
        layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
        layer_norm_affline=True,
    ).to(device)
    oracle_forecaster.load_state_dict(torch.load(
        os.path.join(path_model_save, f"forecaster_exp{args.expid}_run{runid}.pth"),
        weights_only=True,
    ))
    oracle_forecaster.eval()

    csci_model.eval()
    forecaster.eval()

    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    run_mae_all = []   # [split_runs, seq_out_len]
    run_rmse_all = []
    run_spectral_all = []  # spectral metrics per split

    all_idx_tensor = torch.arange(args.num_nodes).to(device)

    for split_run in range(args.random_node_idx_split_runs):
        idx_current_nodes = get_node_random_idx_split(
            args, args.num_nodes,
            args.lower_limit_random_node_selections,
            args.upper_limit_random_node_selections,
        )
        obs_idx = torch.tensor(idx_current_nodes, dtype=torch.long).to(device)
        miss_idx_np = np.setdiff1d(np.arange(args.num_nodes), idx_current_nodes)
        miss_idx = torch.tensor(miss_idx_np, dtype=torch.long).to(device)

        # Per-split realy subsets
        realy_obs = realy[:, idx_current_nodes, :]
        realy_miss = realy[:, miss_idx_np, :]

        outputs_obs = []      # CSCI → obs vars
        outputs_miss = []     # CSCI → miss vars
        outputs_oracle = []   # Forecaster on full input → obs vars
        split_V_miss_hat = []  # for spectral metrics
        split_V_miss_true = []
        split_sigma = []

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testx_masked = zero_out_remaining_input(testx.clone(), idx_current_nodes, args.device)

            with torch.no_grad():
                # CSCI path
                fc_input, attn_bias, V_miss_hat, sigma = csci_model(
                    testx_masked, obs_idx, miss_idx, x_full_unmasked=testx
                )

                # Collect spectral data for metrics
                if len(miss_idx) > 0:
                    V_miss_true = torch.fft.rfft(
                        testx[:, 0, miss_idx, :].transpose(1, 2), dim=1, norm='ortho'
                    )
                    split_V_miss_hat.append(V_miss_hat)
                    split_V_miss_true.append(V_miss_true)
                    split_sigma.append(sigma)
                preds_all = forecaster(fc_input, idx=all_idx_tensor, args=args)
                preds_all = preds_all.transpose(1, 3)[:, 0, :, :]

                outputs_obs.append(preds_all[:, idx_current_nodes, :])
                outputs_miss.append(preds_all[:, miss_idx_np, :])

                # Oracle path: Stage 1 forecaster on unmasked full input
                oracle_preds = oracle_forecaster(testx, idx=all_idx_tensor, args=args)
                oracle_preds = oracle_preds.transpose(1, 3)[:, 0, :, :]
                outputs_oracle.append(oracle_preds[:, idx_current_nodes, :])

        yhat_obs = torch.cat(outputs_obs, dim=0)[:realy_obs.size(0)]
        yhat_miss = torch.cat(outputs_miss, dim=0)[:realy_miss.size(0)]
        yhat_oracle = torch.cat(outputs_oracle, dim=0)[:realy_obs.size(0)]

        # Compute per-horizon metrics for all 3 settings
        obs_mae, obs_rmse = [], []
        miss_mae, miss_rmse = [], []
        oracle_mae, oracle_rmse = [], []

        for h in range(args.seq_out_len):
            # Obs (CSCI → observed vars, VIDA-compatible)
            m, r = metric(scaler.inverse_transform(yhat_obs[:, :, h]), realy_obs[:, :, h])
            obs_mae.append(m); obs_rmse.append(r)

            # Miss (CSCI → missing vars, reconstruction quality)
            m, r = metric(scaler.inverse_transform(yhat_miss[:, :, h]), realy_miss[:, :, h])
            miss_mae.append(m); miss_rmse.append(r)

            # Oracle (full input → observed vars, upper bound)
            m, r = metric(scaler.inverse_transform(yhat_oracle[:, :, h]), realy_obs[:, :, h])
            oracle_mae.append(m); oracle_rmse.append(r)

        # Spectral metrics for this split
        spectral_metrics = {}
        if split_V_miss_hat:
            all_V_hat = torch.cat(split_V_miss_hat, dim=0)
            all_V_true = torch.cat(split_V_miss_true, dim=0)
            all_sigma = torch.cat(split_sigma, dim=0)
            spectral_metrics = evaluate_spectral_metrics(all_V_hat, all_V_true, all_sigma)

        run_mae_all.append(obs_mae)
        run_rmse_all.append(obs_rmse)
        run_spectral_all.append(spectral_metrics)
        tracker.log_eval_split(split_run, obs_mae, obs_rmse,
                               miss_mae=miss_mae, miss_rmse=miss_rmse,
                               oracle_mae=oracle_mae, oracle_rmse=oracle_rmse,
                               spectral_metrics=spectral_metrics)

        if split_run % 10 == 0:
            crps_str = f"  CRPS: {spectral_metrics.get('crps', 0):.4f}" if spectral_metrics else ""
            print(f"  Split {split_run:3d} | "
                  f"obsMAE: {np.mean(obs_mae):.4f}  missMAE: {np.mean(miss_mae):.4f}  "
                  f"oracleMAE: {np.mean(oracle_mae):.4f} | nodes: {len(idx_current_nodes)}"
                  f"{crps_str}")

    # Aggregate spectral metrics
    if run_spectral_all and run_spectral_all[0]:
        metric_keys = run_spectral_all[0].keys()
        avg_spectral = {}
        for k in metric_keys:
            vals = [s[k] for s in run_spectral_all if k in s]
            avg_spectral[k] = {'mean': np.mean(vals), 'std': np.std(vals)}
        print(f"\n  Spectral Metrics (avg over {len(run_spectral_all)} splits):")
        for k, v in avg_spectral.items():
            print(f"    {k}: {v['mean']:.4f} +- {v['std']:.4f}")
        tracker.spectral_summary = avg_spectral

    # Save results
    tracker.save()
    tracker.save_summary()
    tracker.print_summary()

    return run_mae_all, run_rmse_all


if __name__ == "__main__":
    starttime = datetime.datetime.now()

    args = build_args()
    setup_seed(args.seed)
    torch.set_num_threads(3)

    print(f"\n{'='*60}")
    print(f"  CSCI Experiment: {args.data} | Device: {args.device}")
    print(f"  Seed: {args.seed} | ExpID: {args.expid}")
    print(f"{'='*60}\n")
    print(args)

    # Collect results across all runs (VIDA protocol: runs × splits)
    all_mae = []
    all_rmse = []

    for run_i in range(args.runs):
        print(f"\n>>> Run {run_i + 1}/{args.runs}")
        run_mae, run_rmse = main(args, run_i)
        all_mae.extend(run_mae)
        all_rmse.extend(run_rmse)

    # Aggregate across all runs × splits (VIDA-compatible reporting)
    all_mae = np.array(all_mae)   # [total_splits, seq_out_len]
    all_rmse = np.array(all_rmse)

    amae = np.mean(all_mae, axis=0)    # [seq_out_len]
    armse = np.mean(all_rmse, axis=0)
    smae = np.std(all_mae, axis=0)
    srmse = np.std(all_rmse, axis=0)

    print(f"\n\n{'='*60}")
    print(f"  Final Results ({args.runs} runs × {args.random_node_idx_split_runs} splits)")
    print(f"{'='*60}")
    for i in range(args.seq_out_len):
        print(f"  h{i+1:2d}  MAE: {amae[i]:.4f} +- {smae[i]:.4f}  |  "
              f"RMSE: {armse[i]:.4f} +- {srmse[i]:.4f}")
    print(f"{'─'*60}")
    print(f"  Final  MAE: {np.mean(amae):.4f} +- {np.mean(smae):.4f}  |  "
          f"RMSE: {np.mean(armse):.4f} +- {np.mean(srmse):.4f}")
    print(f"  Final  MAE: {np.mean(amae):.2f}({np.mean(smae):.2f})  |  "
          f"RMSE: {np.mean(armse):.2f}({np.mean(srmse):.2f})")
    print(f"{'='*60}")

    endtime = datetime.datetime.now()
    print(f"\nTotal time: {(endtime - starttime).seconds}s")
