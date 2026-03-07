#!/bin/bash
# Full experiment — S init from data + lambda_reg=0.1 + V_miss_hat clamp
# fc_epochs=100, csci_epochs=100, 100 splits, 1 run

PYTHON=/home/sheda7788/.conda/envs/tslib_env/bin/python
EXPID=6
SEED=3407
FC_EP=100
CSCI_EP=100
SPLITS=100
LAMBDA=0.1

cd /home/sheda7788/project/VSF_CSCI
mkdir -p logs/train logs/eval

echo "=== Full Experiment (ExpID=$EXPID) ==="
echo "fc_epochs=$FC_EP | csci_epochs=$CSCI_EP | splits=$SPLITS | lambda_reg=$LAMBDA"
echo "Start: $(date)"
echo ""

# ── GPU 0: METR-LA ──
echo "[GPU 0] METR-LA..."
nohup $PYTHON -u main_csci.py \
    --data data/METR-LA \
    --adj_data data/sensor_graph/adj_mx.pkl \
    --device cuda:0 \
    --seed $SEED --expid $EXPID \
    --model_name mtgnn --in_dim 2 --d_model 64 \
    --fc_epochs $FC_EP --csci_epochs $CSCI_EP \
    --batch_size 64 --runs 1 \
    --random_node_idx_split_runs $SPLITS \
    --step_size1 2500 --patience 20 \
    --lambda_reg $LAMBDA \
    --n_rounds 2 --use_curriculum True \
    --print_every 50 \
    > logs/train/full_METR-LA.log 2>&1 &
echo "  PID: $!"

# ── GPU 1: SOLAR ──
echo "[GPU 1] SOLAR..."
nohup $PYTHON -u main_csci.py \
    --data data/SOLAR \
    --device cuda:1 \
    --seed $SEED --expid $EXPID \
    --model_name mtgnn --in_dim 1 --d_model 64 \
    --fc_epochs $FC_EP --csci_epochs $CSCI_EP \
    --batch_size 64 --runs 1 \
    --random_node_idx_split_runs $SPLITS \
    --step_size1 2500 --patience 20 \
    --lambda_reg $LAMBDA \
    --n_rounds 2 --use_curriculum True \
    --print_every 50 \
    > logs/train/full_SOLAR.log 2>&1 &
echo "  PID: $!"

# ── GPU 2: ECG → TRAFFIC (sequential) ──
echo "[GPU 2] ECG → TRAFFIC..."
nohup bash -c "
    $PYTHON -u main_csci.py \
        --data data/ECG \
        --device cuda:2 \
        --seed $SEED --expid $EXPID \
        --model_name mtgnn --in_dim 1 --d_model 64 \
        --fc_epochs $FC_EP --csci_epochs $CSCI_EP \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 400 --patience 20 \
        --lambda_reg $LAMBDA \
        --n_rounds 2 --use_curriculum True \
        --print_every 50 \
    && echo '--- ECG done, starting TRAFFIC ---' && \
    $PYTHON -u main_csci.py \
        --data data/TRAFFIC \
        --device cuda:2 \
        --seed $SEED --expid $EXPID \
        --model_name mtgnn --in_dim 1 --d_model 64 \
        --fc_epochs $FC_EP --csci_epochs $CSCI_EP \
        --batch_size 32 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 1000 --patience 20 \
        --lambda_reg $LAMBDA \
        --n_rounds 2 --use_curriculum True \
        --print_every 50
" > logs/train/full_ECG_TRAFFIC.log 2>&1 &
echo "  PID: $!"

echo ""
echo "=== Launched ==="
echo "Monitor:"
echo "  tail -f logs/train/full_METR-LA.log"
echo "  tail -f logs/train/full_SOLAR.log"
echo "  tail -f logs/train/full_ECG_TRAFFIC.log"
echo ""
echo "Quick status:"
echo "  grep 'Final\|Stage\|Epoch.*Time' logs/train/full_*.log | tail -20"
