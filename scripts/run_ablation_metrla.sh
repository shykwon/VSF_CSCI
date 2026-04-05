#!/bin/bash
# Ablation experiments on METR-LA (single seed for quick estimation)
# GPU 1, seed=0, each experiment sequential
# ExpID 17: A — obs-only forecast loss
# ExpID 18: B — consistency loss (gamma=0.1)
# ExpID 19: C — Stage 1.5 masking 15% obs (=85% missing)
# ExpID 20: D — n_rounds=2 (hierarchical propagation)

cd /home/sheda7788/VSF_CSCI

COMMON="--device cuda:1 --seed 0 --model_name mtgnn --in_dim 2 --num_nodes 207 \
    --fc_epochs 100 --s_epochs 20 --cvfa_epochs 100 \
    --s_rank 16 --lambda_reg 0.1 \
    --batch_size 64 --runs 1 \
    --random_node_idx_split_runs 100 \
    --step_size1 2500 --patience 20 \
    --alpha_loss 0.7 --beta_loss 0.3 \
    --print_every 50"

# ── Baseline (ExpID 16): same as original for comparison ──
echo "=============================================="
echo ">>> [Baseline] ExpID=16 start $(date)"
echo "=============================================="
/opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py --data data/METR-LA --expid 16 $COMMON
echo ">>> [Baseline] done $(date)"
echo ""

# ── ExpA (ExpID 17): obs-only forecast loss ──
echo "=============================================="
echo ">>> [ExpA] obs-only loss, ExpID=17 start $(date)"
echo "=============================================="
/opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py --data data/METR-LA --expid 17 $COMMON \
    --obs_only_loss True
echo ">>> [ExpA] done $(date)"
echo ""

# ── ExpB (ExpID 18): consistency loss ──
echo "=============================================="
echo ">>> [ExpB] consistency loss, ExpID=18 start $(date)"
echo "=============================================="
/opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py --data data/METR-LA --expid 18 $COMMON \
    --gamma_loss 0.1
echo ">>> [ExpB] done $(date)"
echo ""

# ── ExpC (ExpID 19): Stage 1.5 masking aligned to 15% obs ──
echo "=============================================="
echo ">>> [ExpC] S-train obs_ratio=0.15, ExpID=19 start $(date)"
echo "=============================================="
/opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py --data data/METR-LA --expid 19 $COMMON \
    --s_train_obs_ratio 0.15
echo ">>> [ExpC] done $(date)"
echo ""

# ── ExpD (ExpID 20): n_rounds=2 hierarchical propagation ──
echo "=============================================="
echo ">>> [ExpD] n_rounds=2, ExpID=20 start $(date)"
echo "=============================================="
/opt/anaconda3/envs/comet/bin/python3 -u main_cvfa.py --data data/METR-LA --expid 20 $COMMON \
    --n_rounds 2
echo ">>> [ExpD] done $(date)"
echo ""

echo "=============================================="
echo ">>> ALL ablation experiments DONE $(date)"
echo "=============================================="
