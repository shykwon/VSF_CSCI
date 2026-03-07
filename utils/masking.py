# VSF masking utilities — aligned with VIDA protocol
# Reference: vida-vsf/VIDA/util.py

import numpy as np
import math
import torch


def get_node_random_idx_split(args, num_nodes, lb, ub):
    """
    Randomly select a subset of node indices (VIDA-compatible).

    Args:
        num_nodes: total number of variables
        lb, ub: lower/upper bound percentage for subset size
    Returns:
        current_node_idxs: numpy array of selected node indices
    """
    count_percent = np.random.choice(np.arange(lb, ub + 1), size=1, replace=False)[0]
    count = math.ceil(num_nodes * (count_percent / 100))
    all_node_idxs = np.arange(num_nodes)
    current_node_idxs = np.random.choice(all_node_idxs, size=count, replace=False)
    return current_node_idxs


def zero_out_remaining_input(testx, idx_current_nodes, device):
    """
    Zero-fill unobserved variables (VIDA-compatible).

    Args:
        testx: [B, in_dim, num_nodes, seq_len]
        idx_current_nodes: observed node indices
    Returns:
        masked input with unobserved variables set to 0.0
    """
    zero_val_mask = torch.ones_like(testx).bool()
    zero_val_mask[:, :, idx_current_nodes, :] = False
    inps = testx.masked_fill_(zero_val_mask, 0.0)
    return inps


def get_curriculum_mask_ratio(epoch, max_epoch=120, max_ratio=0.85):
    """
    Progressive masking curriculum for CSCI training.

    Schedule:
        Epoch 1~25%:   miss rate up to 30%
        Epoch 25~50%:  miss rate up to 60%
        Epoch 50~75%:  miss rate up to 85%
        Epoch 75~100%: random uniform sampling [0, max_ratio]

    Args:
        epoch: current epoch (0-indexed)
        max_epoch: total number of epochs
        max_ratio: maximum missing ratio
    Returns:
        mask_ratio: float in [0, max_ratio]
    """
    if epoch < max_epoch * 0.25:
        return np.random.uniform(0, 0.30)
    elif epoch < max_epoch * 0.50:
        return np.random.uniform(0, 0.60)
    elif epoch < max_epoch * 0.75:
        return np.random.uniform(0, 0.85)
    else:
        return np.random.uniform(0, max_ratio)


def get_idx_subset_from_idx_all_nodes(idx_all_nodes, mask_ratio=0.15):
    """
    Select a random subset from given node indices (VIDA-compatible).

    Args:
        idx_all_nodes: tensor of all node indices
        mask_ratio: fraction of nodes to select
    Returns:
        numpy array of selected subset indices
    """
    idx_all_nodes = idx_all_nodes.cpu().detach().numpy()
    length_subset = math.ceil(len(idx_all_nodes) * mask_ratio)
    idx_subset = np.random.choice(idx_all_nodes, size=length_subset, replace=False)
    return idx_subset
