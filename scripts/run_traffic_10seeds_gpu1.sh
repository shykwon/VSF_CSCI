#!/bin/bash
# TRAFFIC 10-seed experiment on GPU 1 (A30 24GB)
# batch_size=64, sequential seed 0-9

cd /home/sheda7788/VSF_CSCI

for SEED in $(seq 0 9); do
    echo "=============================================="
    echo ">>> TRAFFIC seed=$SEED start $(date)"
    echo "=============================================="
    python3 -u main_cvfa.py \
        --data data/TRAFFIC --device cuda:1 \
        --seed $SEED --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 862 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 1000 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> TRAFFIC seed=$SEED done $(date)"
    echo ""
done

echo "=============================================="
echo ">>> ALL 10 seeds DONE $(date)"
echo "=============================================="
