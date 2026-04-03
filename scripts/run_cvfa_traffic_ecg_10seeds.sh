#!/bin/bash
# CVFA 10-seed experiments: TRAFFIC + ECG
# GPU 0: ECG seed 0-9 (가벼움, ~140 nodes) → TRAFFIC seed 0-3
# GPU 1: (METR-LA 완료 대기 후) TRAFFIC seed 4-6
# GPU 2: (METR-LA 완료 대기 후) TRAFFIC seed 7-9
#
# ECG: step_size1=400, 140 nodes
# TRAFFIC: step_size1=1000, 862 nodes

cd /home/sheda7788/VSF_CSCI

run_ecg() {
    local GPU=$1
    local SEED=$2
    echo ">>> ECG seed=$SEED start $(date)"
    python3 -u main_cvfa.py \
        --data data/ECG --device cuda:$GPU \
        --seed $SEED --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 140 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 64 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 400 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> ECG seed=$SEED done $(date)"
}

run_traffic() {
    local GPU=$1
    local SEED=$2
    echo ">>> TRAFFIC seed=$SEED start $(date)"
    python3 -u main_cvfa.py \
        --data data/TRAFFIC --device cuda:$GPU \
        --seed $SEED --expid 16 \
        --model_name mtgnn --in_dim 1 --num_nodes 862 \
        --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
        --s_rank 16 --lambda_reg 0.1 \
        --batch_size 32 --runs 1 \
        --random_node_idx_split_runs 100 \
        --step_size1 1000 --patience 20 \
        --n_rounds 1 --alpha_loss 0.7 --beta_loss 0.3 \
        --print_every 50
    echo ">>> TRAFFIC seed=$SEED done $(date)"
}

export -f run_ecg run_traffic
