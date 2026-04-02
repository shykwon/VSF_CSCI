#!/bin/bash
# CVFA 10-seed runs: SOLAR + METR-LA — sequential per GPU
# METR-LA: in_dim=1 (VIDA-compatible, no time_of_day)
# GPU 0: SOLAR 0-3 → METR-LA 0-2
# GPU 1: SOLAR 4-6 → METR-LA 3-5
# GPU 2: SOLAR 7-9 → METR-LA 6-9

cd /home/sheda7788/VSF_CSCI
mkdir -p logs/train

echo "=== CVFA 10-seed: SOLAR + METR-LA (in_dim=1) ==="
echo "Start: $(date)"

# ── GPU 0: SOLAR 0-3 → METR-LA 0-2 ──
nohup bash -c '
for S in 0 1 2 3; do
    echo ">>> SOLAR seed=$S start $(date)"
    python3 -u main_cvfa.py \
        --data data/SOLAR --device cuda:0 \
        --seed $S --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 137 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> SOLAR seed=$S done $(date)"
done
for S in 0 1 2; do
    echo ">>> METR-LA seed=$S start $(date)"
    python3 -u main_cvfa.py \
        --data data/METR-LA \
        --adj_data data/sensor_graph/adj_mx.pkl \
        --device cuda:0 \
        --seed $S --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 207 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> METR-LA seed=$S done $(date)"
done
' > logs/train/cvfa_gpu0.log 2>&1 &
echo "[GPU 0] SOLAR 0-3 → METR-LA 0-2 | PID: $!"

# ── GPU 1: SOLAR 4-6 → METR-LA 3-5 ──
nohup bash -c '
for S in 4 5 6; do
    echo ">>> SOLAR seed=$S start $(date)"
    python3 -u main_cvfa.py \
        --data data/SOLAR --device cuda:1 \
        --seed $S --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 137 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> SOLAR seed=$S done $(date)"
done
for S in 3 4 5; do
    echo ">>> METR-LA seed=$S start $(date)"
    python3 -u main_cvfa.py \
        --data data/METR-LA \
        --adj_data data/sensor_graph/adj_mx.pkl \
        --device cuda:1 \
        --seed $S --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 207 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> METR-LA seed=$S done $(date)"
done
' > logs/train/cvfa_gpu1.log 2>&1 &
echo "[GPU 1] SOLAR 4-6 → METR-LA 3-5 | PID: $!"

# ── GPU 2: SOLAR 7-9 → METR-LA 6-9 ──
nohup bash -c '
for S in 7 8 9; do
    echo ">>> SOLAR seed=$S start $(date)"
    python3 -u main_cvfa.py \
        --data data/SOLAR --device cuda:2 \
        --seed $S --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 137 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> SOLAR seed=$S done $(date)"
done
for S in 6 7 8 9; do
    echo ">>> METR-LA seed=$S start $(date)"
    python3 -u main_cvfa.py \
        --data data/METR-LA \
        --adj_data data/sensor_graph/adj_mx.pkl \
        --device cuda:2 \
        --seed $S --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 207 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> METR-LA seed=$S done $(date)"
done
' > logs/train/cvfa_gpu2.log 2>&1 &
echo "[GPU 2] SOLAR 7-9 → METR-LA 6-9 | PID: $!"

echo ""
echo "=== Launched ==="
echo "SOLAR ~3hr/seed, METR-LA ~2.3hr/seed"
echo "GPU 0: 4+3=~19hr | GPU 1: 3+3=~16hr | GPU 2: 3+4=~18hr"
echo ""
echo "Monitor:"
echo "  grep 'done\|start' logs/train/cvfa_gpu*.log"
