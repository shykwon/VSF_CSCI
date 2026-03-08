#!/bin/bash
# Full experiment — projected mode (obs original + miss projection)
# fc_epochs=100, csci_epochs=100, s_epochs=20, 100 splits, 1 run

PYTHON=/home/sheda7788/.conda/envs/tslib_env/bin/python
EXPID=10
SEED=3407
FC_EP=100
CSCI_EP=100
S_EP=20
S_RANK=0
SPLITS=100
LAMBDA=0.1

cd /home/sheda7788/project/VSF_CSCI
mkdir -p logs/train logs/eval

echo "=== Full Experiment (ExpID=$EXPID) ==="
echo "fc=$FC_EP | s=$S_EP | csci=$CSCI_EP | rank=$S_RANK | lambda=$LAMBDA | splits=$SPLITS"
echo "Start: $(date)"
echo ""

# ── GPU 0: ECG → METR-LA (sequential) ──
echo "[GPU 0] ECG → METR-LA..."
nohup bash -c "
    $PYTHON -u main_csci.py \
        --data data/ECG \
        --device cuda:0 \
        --seed $SEED --expid $EXPID \
        --model_name mtgnn --in_dim 1 --d_model 64 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --lambda_reg $LAMBDA \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 400 --patience 20 \
        --n_rounds 2 --use_curriculum True \
        --print_every 50 \
    && echo '--- ECG done, starting METR-LA ---' && \
    $PYTHON -u main_csci.py \
        --data data/METR-LA \
        --adj_data data/sensor_graph/adj_mx.pkl \
        --device cuda:0 \
        --seed $SEED --expid $EXPID \
        --model_name mtgnn --in_dim 2 --d_model 64 \
        --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
        --s_rank $S_RANK --lambda_reg $LAMBDA \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs $SPLITS \
        --step_size1 2500 --patience 20 \
        --n_rounds 2 --use_curriculum True \
        --print_every 50
" > logs/train/full_ECG_METR-LA.log 2>&1 &
echo "  PID: $!"

# ── GPU 1: SOLAR ──
echo "[GPU 1] SOLAR..."
nohup $PYTHON -u main_csci.py \
    --data data/SOLAR \
    --device cuda:1 \
    --seed $SEED --expid $EXPID \
    --model_name mtgnn --in_dim 1 --d_model 64 \
    --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
    --s_rank $S_RANK --lambda_reg $LAMBDA \
    --batch_size 64 --runs 1 \
    --random_node_idx_split_runs $SPLITS \
    --step_size1 2500 --patience 20 \
    --n_rounds 2 --use_curriculum True \
    --print_every 50 \
    > logs/train/full_SOLAR.log 2>&1 &
echo "  PID: $!"

# ── GPU 2: TRAFFIC ──
echo "[GPU 2] TRAFFIC..."
nohup $PYTHON -u main_csci.py \
    --data data/TRAFFIC \
    --device cuda:2 \
    --seed $SEED --expid $EXPID \
    --model_name mtgnn --in_dim 1 --d_model 64 \
    --fc_epochs $FC_EP --s_epochs $S_EP --csci_epochs $CSCI_EP \
    --s_rank $S_RANK --lambda_reg $LAMBDA \
    --batch_size 32 --runs 1 \
    --random_node_idx_split_runs $SPLITS \
    --step_size1 1000 --patience 20 \
    --n_rounds 2 --use_curriculum True \
    --print_every 50 \
    > logs/train/full_TRAFFIC.log 2>&1 &
echo "  PID: $!"

echo ""
echo "=== Launched ==="
echo "Monitor:"
echo "  tail -f logs/train/full_ECG_METR-LA.log"
echo "  tail -f logs/train/full_SOLAR.log"
echo "  tail -f logs/train/full_TRAFFIC.log"
echo ""
echo "Quick status:"
echo "  grep 'Final\|Stage\|Epoch.*Time' logs/train/full_*.log | tail -20"
