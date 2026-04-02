#!/bin/bash
# Exp13: alpha=0.4, beta=0.5, gamma=0.1 (spectral loss 강화)
# Exp14: lambda_reg=0.05 (Wiener filter 정규화 완화)
# Exp15: d_model=128 (METR-LA, TRAFFIC only — 임베딩 용량 증가)
# 모두 embedding mode (exp8 base)

PYTHON=/home/sheda7788/.conda/envs/tslib_env/bin/python
SEED=3407
FC_EP=100
CSCI_EP=100
S_EP=20
S_RANK=0
SPLITS=100

cd /home/sheda7788/project/VSF_CSCI
mkdir -p logs/train

echo "=== Exp13/14/15 Parameter Tuning (embedding mode base) ==="
echo "Start: $(date)"
echo ""

# ── GPU 0: ECG exp13 → exp14 → METR-LA exp13 → exp14 → exp15 ──
echo "[GPU 0] ECG exp13→14, METR-LA exp13→14→15..."
nohup bash -c "
    echo '=== ECG exp13 (alpha=0.4 beta=0.5) ===' && \
    $PYTHON -u main_csci.py \
        --data data/ECG --device cuda:0 \
        --seed $SEED --expid 13 \
        --head_mode embedding --in_dim 1 --d_model 64 \
        --alpha_loss 0.4 --beta_loss 0.5 --gamma_loss 0.1 \
        --lambda_reg 0.1 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 400 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50 \
    && echo '=== ECG exp14 (lambda=0.05) ===' && \
    $PYTHON -u main_csci.py \
        --data data/ECG --device cuda:0 \
        --seed $SEED --expid 14 \
        --head_mode embedding --in_dim 1 --d_model 64 \
        --alpha_loss 0.6 --beta_loss 0.3 --gamma_loss 0.1 \
        --lambda_reg 0.05 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 400 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50 \
    && echo '=== METR-LA exp13 (alpha=0.4 beta=0.5) ===' && \
    $PYTHON -u main_csci.py \
        --data data/METR-LA --adj_data data/sensor_graph/adj_mx.pkl \
        --device cuda:0 \
        --seed $SEED --expid 13 \
        --head_mode embedding --in_dim 2 --d_model 64 \
        --alpha_loss 0.4 --beta_loss 0.5 --gamma_loss 0.1 \
        --lambda_reg 0.1 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50 \
    && echo '=== METR-LA exp14 (lambda=0.05) ===' && \
    $PYTHON -u main_csci.py \
        --data data/METR-LA --adj_data data/sensor_graph/adj_mx.pkl \
        --device cuda:0 \
        --seed $SEED --expid 14 \
        --head_mode embedding --in_dim 2 --d_model 64 \
        --alpha_loss 0.6 --beta_loss 0.3 --gamma_loss 0.1 \
        --lambda_reg 0.05 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50 \
    && echo '=== METR-LA exp15 (d_model=128) ===' && \
    $PYTHON -u main_csci.py \
        --data data/METR-LA --adj_data data/sensor_graph/adj_mx.pkl \
        --device cuda:0 \
        --seed $SEED --expid 15 \
        --head_mode embedding --in_dim 2 --d_model 128 \
        --alpha_loss 0.6 --beta_loss 0.3 --gamma_loss 0.1 \
        --lambda_reg 0.1 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50
" > logs/train/full_gpu0_exp13_14_15.log 2>&1 &
echo "  PID: $!"

# ── GPU 1: SOLAR exp13 → exp14 ──
echo "[GPU 1] SOLAR exp13→14..."
nohup bash -c "
    echo '=== SOLAR exp13 (alpha=0.4 beta=0.5) ===' && \
    $PYTHON -u main_csci.py \
        --data data/SOLAR --device cuda:1 \
        --seed $SEED --expid 13 \
        --head_mode embedding --in_dim 1 --d_model 64 \
        --alpha_loss 0.4 --beta_loss 0.5 --gamma_loss 0.1 \
        --lambda_reg 0.1 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50 \
    && echo '=== SOLAR exp14 (lambda=0.05) ===' && \
    $PYTHON -u main_csci.py \
        --data data/SOLAR --device cuda:1 \
        --seed $SEED --expid 14 \
        --head_mode embedding --in_dim 1 --d_model 64 \
        --alpha_loss 0.6 --beta_loss 0.3 --gamma_loss 0.1 \
        --lambda_reg 0.05 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50
" > logs/train/full_gpu1_exp13_14.log 2>&1 &
echo "  PID: $!"

# ── GPU 2: TRAFFIC exp13 → exp14 → exp15 ──
echo "[GPU 2] TRAFFIC exp13→14→15..."
nohup bash -c "
    echo '=== TRAFFIC exp13 (alpha=0.4 beta=0.5) ===' && \
    $PYTHON -u main_csci.py \
        --data data/TRAFFIC --device cuda:2 \
        --seed $SEED --expid 13 \
        --head_mode embedding --in_dim 1 --d_model 64 \
        --alpha_loss 0.4 --beta_loss 0.5 --gamma_loss 0.1 \
        --lambda_reg 0.1 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 32 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 1000 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50 \
    && echo '=== TRAFFIC exp14 (lambda=0.05) ===' && \
    $PYTHON -u main_csci.py \
        --data data/TRAFFIC --device cuda:2 \
        --seed $SEED --expid 14 \
        --head_mode embedding --in_dim 1 --d_model 64 \
        --alpha_loss 0.6 --beta_loss 0.3 --gamma_loss 0.1 \
        --lambda_reg 0.05 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 32 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 1000 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50 \
    && echo '=== TRAFFIC exp15 (d_model=128) ===' && \
    $PYTHON -u main_csci.py \
        --data data/TRAFFIC --device cuda:2 \
        --seed $SEED --expid 15 \
        --head_mode embedding --in_dim 1 --d_model 128 \
        --alpha_loss 0.6 --beta_loss 0.3 --gamma_loss 0.1 \
        --lambda_reg 0.1 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --batch_size 32 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 1000 --patience 20 \
        --n_rounds 2 --use_curriculum True --print_every 50
" > logs/train/full_gpu2_exp13_14_15.log 2>&1 &
echo "  PID: $!"

echo ""
echo "=== Launched ==="
echo "Monitor:"
echo "  tail -f logs/train/full_gpu0_exp13_14_15.log"
echo "  tail -f logs/train/full_gpu1_exp13_14.log"
echo "  tail -f logs/train/full_gpu2_exp13_14_15.log"
