#!/bin/bash
# Exp11: Projected + MLP miss_proj
# Exp12: Embedding + obs_residual + smart init
# GPU 0: ECG exp11 → METR-LA exp11
# GPU 1: SOLAR exp11 → SOLAR exp12
# GPU 2: TRAFFIC exp11 → TRAFFIC exp12

PYTHON=/home/sheda7788/.conda/envs/tslib_env/bin/python
SEED=3407
FC_EP=100
CSCI_EP=100
S_EP=20
S_RANK=0
SPLITS=100
LAMBDA=0.1

cd /home/sheda7788/project/VSF_CSCI
mkdir -p logs/train logs/eval

echo "=== Exp11 (projected+MLP) & Exp12 (embedding+residual) ==="
echo "Start: $(date)"
echo ""

# ── GPU 0: ECG exp11 → METR-LA exp11 (sequential) ──
echo "[GPU 0] ECG exp11 → METR-LA exp11..."
nohup bash -c "
    $PYTHON -u main_csci.py \
        --data data/ECG \
        --device cuda:0 \
        --seed $SEED --expid 11 \
        --model_name mtgnn --in_dim 1 --d_model 64 \
        --head_mode projected --miss_proj_type mlp \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --lambda_reg $LAMBDA \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 400 --patience 20 \
        --n_rounds 2 --use_curriculum True \
        --print_every 50 \
    && echo '--- ECG exp11 done, starting METR-LA exp11 ---' && \
    $PYTHON -u main_csci.py \
        --data data/METR-LA \
        --adj_data data/sensor_graph/adj_mx.pkl \
        --device cuda:0 \
        --seed $SEED --expid 11 \
        --model_name mtgnn --in_dim 2 --d_model 64 \
        --head_mode projected --miss_proj_type mlp \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --lambda_reg $LAMBDA \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True \
        --print_every 50
" > logs/train/full_ECG_METR-LA_exp11.log 2>&1 &
echo "  PID: $!"

# ── GPU 1: SOLAR exp11 → SOLAR exp12 (sequential) ──
echo "[GPU 1] SOLAR exp11 → SOLAR exp12..."
nohup bash -c "
    $PYTHON -u main_csci.py \
        --data data/SOLAR \
        --device cuda:1 \
        --seed $SEED --expid 11 \
        --model_name mtgnn --in_dim 1 --d_model 64 \
        --head_mode projected --miss_proj_type mlp \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --lambda_reg $LAMBDA \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True \
        --print_every 50 \
    && echo '--- SOLAR exp11 done, starting SOLAR exp12 ---' && \
    $PYTHON -u main_csci.py \
        --data data/SOLAR \
        --device cuda:1 \
        --seed $SEED --expid 12 \
        --model_name mtgnn --in_dim 1 --d_model 64 \
        --head_mode embedding --use_obs_residual True \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --lambda_reg $LAMBDA \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True \
        --print_every 50
" > logs/train/full_SOLAR_exp11_12.log 2>&1 &
echo "  PID: $!"

# ── GPU 2: TRAFFIC exp11 → TRAFFIC exp12 (sequential) ──
echo "[GPU 2] TRAFFIC exp11 → TRAFFIC exp12..."
nohup bash -c "
    $PYTHON -u main_csci.py \
        --data data/TRAFFIC \
        --device cuda:2 \
        --seed $SEED --expid 11 \
        --model_name mtgnn --in_dim 1 --d_model 64 \
        --head_mode projected --miss_proj_type mlp \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --lambda_reg $LAMBDA \
        --batch_size 32 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 1000 --patience 20 \
        --n_rounds 2 --use_curriculum True \
        --print_every 50 \
    && echo '--- TRAFFIC exp11 done, starting TRAFFIC exp12 ---' && \
    $PYTHON -u main_csci.py \
        --data data/TRAFFIC \
        --device cuda:2 \
        --seed $SEED --expid 12 \
        --model_name mtgnn --in_dim 1 --d_model 64 \
        --head_mode embedding --use_obs_residual True \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --lambda_reg $LAMBDA \
        --batch_size 32 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 1000 --patience 20 \
        --n_rounds 2 --use_curriculum True \
        --print_every 50
" > logs/train/full_TRAFFIC_exp11_12.log 2>&1 &
echo "  PID: $!"

echo ""
echo "=== Launched ==="
echo "Monitor:"
echo "  tail -f logs/train/full_ECG_METR-LA_exp11.log"
echo "  tail -f logs/train/full_SOLAR_exp11_12.log"
echo "  tail -f logs/train/full_TRAFFIC_exp11_12.log"
echo ""
echo "Quick status:"
echo "  grep 'Final\|Stage\|Epoch.*Time' logs/train/full_*exp1[12]*.log | tail -20"
