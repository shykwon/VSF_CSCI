#!/bin/bash
# CVFA SOLAR: 10 seeds (0-9) across 3 GPUs
# GPU 0: seeds 0,1,2,3 | GPU 1: seeds 4,5,6 | GPU 2: seeds 7,8,9

EXPID=16
FC_EP=100
S_EP=20
CVFA_EP=100
SPLITS=100

cd /home/sheda7788/VSF_CSCI
mkdir -p logs/train

echo "=== CVFA SOLAR 10-seed run (ExpID=$EXPID) ==="
echo "Start: $(date)"

# ── GPU 0: seeds 0,1,2,3 ──
echo "[GPU 0] seeds 0,1,2,3"
nohup bash -c '
for SEED in 0 1 2 3; do
    echo ">>> GPU 0: seed=$SEED start $(date)"
    python3 -u main_cvfa.py \
        --data data/SOLAR --device cuda:0 \
        --seed $SEED --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 137 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> GPU 0: seed=$SEED done $(date)"
done
' > logs/train/cvfa_SOLAR_10seeds_gpu0.log 2>&1 &
echo "  PID: $!"

# ── GPU 1: seeds 4,5,6 ──
echo "[GPU 1] seeds 4,5,6"
nohup bash -c '
for SEED in 4 5 6; do
    echo ">>> GPU 1: seed=$SEED start $(date)"
    python3 -u main_cvfa.py \
        --data data/SOLAR --device cuda:1 \
        --seed $SEED --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 137 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> GPU 1: seed=$SEED done $(date)"
done
' > logs/train/cvfa_SOLAR_10seeds_gpu1.log 2>&1 &
echo "  PID: $!"

# ── GPU 2: seeds 7,8,9 ──
echo "[GPU 2] seeds 7,8,9"
nohup bash -c '
for SEED in 7 8 9; do
    echo ">>> GPU 2: seed=$SEED start $(date)"
    python3 -u main_cvfa.py \
        --data data/SOLAR --device cuda:2 \
        --seed $SEED --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 137 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> GPU 2: seed=$SEED done $(date)"
done
' > logs/train/cvfa_SOLAR_10seeds_gpu2.log 2>&1 &
echo "  PID: $!"

echo ""
echo "=== Launched ==="
echo "Monitor:"
echo "  tail -f logs/train/cvfa_SOLAR_10seeds_gpu0.log"
echo "  tail -f logs/train/cvfa_SOLAR_10seeds_gpu1.log"
echo "  tail -f logs/train/cvfa_SOLAR_10seeds_gpu2.log"
echo ""
echo "Quick status:"
echo "  grep 'Final\|seed=.*done\|seed=.*start' logs/train/cvfa_SOLAR_10seeds_gpu*.log"
