#!/bin/bash
# Run CSCI experiments across 3 GPUs (GTX 1080 × 3)
# Each dataset assigned to a separate GPU for parallel execution.
# Uses nohup for background execution (survives session disconnect).

PYTHON=/home/sheda7788/.conda/envs/tslib_env/bin/python
EXPID=1
SEED=3407

echo "=== CSCI Experiment Suite ==="
echo "ExpID: $EXPID | Seed: $SEED"
echo "Starting time: $(date)"
echo ""

mkdir -p logs/train logs/eval

# ── GPU 0: METR-LA ──
echo "[GPU 0] Starting METR-LA..."
nohup $PYTHON -u main_csci.py \
    --data data/METR-LA \
    --adj_data data/sensor_graph/adj_mx.pkl \
    --device cuda:0 \
    --seed $SEED \
    --expid $EXPID \
    --model_name mtgnn \
    --in_dim 1 \
    --d_model 64 \
    --fc_epochs 100 \
    --csci_epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --step_size1 2500 \
    --patience 20 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    > logs/train/METR-LA_exp${EXPID}_seed${SEED}.log 2>&1 &
PID_METRLA=$!
echo "  PID: $PID_METRLA"

# ── GPU 1: SOLAR ──
echo "[GPU 1] Starting SOLAR..."
nohup $PYTHON -u main_csci.py \
    --data data/SOLAR \
    --device cuda:1 \
    --seed $SEED \
    --expid $EXPID \
    --model_name mtgnn \
    --in_dim 1 \
    --d_model 64 \
    --fc_epochs 100 \
    --csci_epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --step_size1 2500 \
    --patience 20 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    > logs/train/SOLAR_exp${EXPID}_seed${SEED}.log 2>&1 &
PID_SOLAR=$!
echo "  PID: $PID_SOLAR"

# ── GPU 2: ECG5000 + TRAFFIC (sequential, share GPU) ──
echo "[GPU 2] Starting ECG5000..."
nohup bash -c "
    $PYTHON -u main_csci.py \
        --data data/ECG5000 \
        --device cuda:2 \
        --seed $SEED \
        --expid $EXPID \
        --model_name mtgnn \
        --in_dim 1 \
        --d_model 64 \
        --fc_epochs 100 \
        --csci_epochs 100 \
        --batch_size 64 \
        --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 400 \
        --patience 20 \
        --learning_rate 0.001 \
        --weight_decay 0.0001 \
    && \
    echo '[GPU 2] ECG5000 done. Starting TRAFFIC...' && \
    $PYTHON -u main_csci.py \
        --data data/TRAFFIC \
        --device cuda:2 \
        --seed $SEED \
        --expid $EXPID \
        --model_name mtgnn \
        --in_dim 1 \
        --d_model 64 \
        --fc_epochs 100 \
        --csci_epochs 100 \
        --batch_size 32 \
        --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 1000 \
        --patience 20 \
        --learning_rate 0.001 \
        --weight_decay 0.0001
" > logs/train/ECG_TRAFFIC_exp${EXPID}_seed${SEED}.log 2>&1 &
PID_GPU2=$!
echo "  PID: $PID_GPU2"

echo ""
echo "=== All experiments launched ==="
echo "Monitor with:"
echo "  tail -f logs/train/METR-LA_exp${EXPID}_seed${SEED}.log"
echo "  tail -f logs/train/SOLAR_exp${EXPID}_seed${SEED}.log"
echo "  tail -f logs/train/ECG_TRAFFIC_exp${EXPID}_seed${SEED}.log"
echo ""
echo "Check status: ps aux | grep main_csci"
