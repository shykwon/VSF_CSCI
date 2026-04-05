#!/bin/bash
# METR-LA 10-seed experiment with best ablation config (B+A combo)
# ExpID 24: gamma=0.1 + obs_only_loss, seeds 0-9, GPU 1

cd /home/sheda7788/VSF_CSCI

COMMON="--device cuda:1 --model_name mtgnn --in_dim 2 --num_nodes 207 \
    --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
    --s_rank 16 --lambda_reg 0.1 \
    --batch_size 64 --runs 1 \
    --random_node_idx_split_runs 100 \
    --step_size1 2500 --patience 20 \
    --alpha_loss 0.7 --beta_loss 0.3 \
    --gamma_loss 0.1 --obs_only_loss True \
    --print_every 50"

for SEED in 0 1 2 3 4 5 6 7 8 9; do
    echo "=============================================="
    echo ">>> METR-LA seed=${SEED}, ExpID=24 start $(date)"
    echo "=============================================="
    /opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py \
        --data data/METR-LA --expid 24 --seed ${SEED} $COMMON
    echo ">>> seed=${SEED} done $(date)"
    echo ""
done

echo "=============================================="
echo ">>> ALL 10-seed METR-LA DONE $(date)"
echo "=============================================="
