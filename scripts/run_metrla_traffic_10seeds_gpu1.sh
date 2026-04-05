#!/bin/bash
# METR-LA + TRAFFIC 10-seed experiments (GPU 1, sequential)
# Config: obs_only_loss + consistency loss (adopted defaults)
# ExpID 25

cd /home/sheda7788/VSF_CSCI

# ══════════════════════════════════════════════════
#  METR-LA: 10 seeds (in_dim=2, num_nodes=207)
# ══════════════════════════════════════════════════
echo "=============================================="
echo ">>> METR-LA 10-seed START $(date)"
echo "=============================================="

for SEED in $(seq 0 9); do
    echo "=============================================="
    echo ">>> METR-LA seed=${SEED} start $(date)"
    echo "=============================================="
    /opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py \
        --data data/METR-LA --expid 25 --device cuda:1 \
        --seed $SEED --model_name mtgnn --in_dim 2 --num_nodes 207 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 2500 --patience 20 \
        --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> METR-LA seed=${SEED} done $(date)"
    echo ""
done

echo "=============================================="
echo ">>> METR-LA 10-seed ALL DONE $(date)"
echo "=============================================="

# ══════════════════════════════════════════════════
#  TRAFFIC: 10 seeds (in_dim=1, num_nodes=862)
# ══════════════════════════════════════════════════
echo ""
echo "=============================================="
echo ">>> TRAFFIC 10-seed START $(date)"
echo "=============================================="

for SEED in $(seq 0 9); do
    echo "=============================================="
    echo ">>> TRAFFIC seed=${SEED} start $(date)"
    echo "=============================================="
    /opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py \
        --data data/TRAFFIC --expid 25 --device cuda:1 \
        --seed $SEED --model_name mtgnn --in_dim 1 --num_nodes 862 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 1000 --patience 20 \
        --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> TRAFFIC seed=${SEED} done $(date)"
    echo ""
done

echo "=============================================="
echo ">>> ALL EXPERIMENTS DONE $(date)"
echo "=============================================="
