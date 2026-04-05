#!/bin/bash
# Ablation round 2: B+A combo + gamma tuning (METR-LA, single seed)
# ExpID 21: B+A (gamma=0.1 + obs_only_loss)
# ExpID 22: B gamma=0.05
# ExpID 23: B gamma=0.2

cd /home/sheda7788/VSF_CSCI

COMMON="--device cuda:1 --seed 0 --model_name mtgnn --in_dim 2 --num_nodes 207 \
    --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
    --s_rank 16 --lambda_reg 0.1 \
    --batch_size 64 --runs 1 \
    --random_node_idx_split_runs 100 \
    --step_size1 2500 --patience 20 \
    --alpha_loss 0.7 --beta_loss 0.3 \
    --print_every 50"

# ── ExpID 21: B+A combo (consistency + obs-only loss) ──
echo "=============================================="
echo ">>> [B+A] gamma=0.1 + obs_only, ExpID=21 start $(date)"
echo "=============================================="
/opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py --data data/METR-LA --expid 21 $COMMON \
    --gamma_loss 0.1 --obs_only_loss True
echo ">>> [B+A] done $(date)"
echo ""

# ── ExpID 22: B gamma=0.05 ──
echo "=============================================="
echo ">>> [B] gamma=0.05, ExpID=22 start $(date)"
echo "=============================================="
/opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py --data data/METR-LA --expid 22 $COMMON \
    --gamma_loss 0.05
echo ">>> [B] gamma=0.05 done $(date)"
echo ""

# ── ExpID 23: B gamma=0.2 ──
echo "=============================================="
echo ">>> [B] gamma=0.2, ExpID=23 start $(date)"
echo "=============================================="
/opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py --data data/METR-LA --expid 23 $COMMON \
    --gamma_loss 0.2
echo ">>> [B] gamma=0.2 done $(date)"
echo ""

echo "=============================================="
echo ">>> ALL round-2 ablation DONE $(date)"
echo "=============================================="
