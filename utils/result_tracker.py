# Experiment result tracking and summary utilities

import json
import os
import numpy as np
from datetime import datetime


class ResultTracker:
    """
    Track and persist experiment results across runs and splits.

    Directory structure:
        results/
        ├── raw/                         # Per-experiment raw results
        │   └── {exp_name}.json
        ├── summary/                     # Aggregated summary tables
        │   └── {exp_name}_summary.json
        └── figures/                     # Plots (loss curves, comparisons)

    Usage:
        tracker = ResultTracker("results", "csci_metrla_exp1")
        tracker.log_train_epoch(epoch, train_loss, val_loss, val_rmse)
        tracker.log_eval_split(split_id, mae_per_horizon, rmse_per_horizon)
        tracker.save()
        tracker.print_summary()
    """

    def __init__(self, result_dir, exp_name):
        self.result_dir = result_dir
        self.exp_name = exp_name
        self.raw_dir = os.path.join(result_dir, "raw")
        self.summary_dir = os.path.join(result_dir, "summary")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

        self.data = {
            "exp_name": exp_name,
            "created_at": datetime.now().isoformat(),
            "config": {},
            "train_history": [],
            "eval_splits": [],
        }

    def set_config(self, args):
        """Save experiment configuration."""
        if hasattr(args, '__dict__'):
            self.data["config"] = {k: str(v) for k, v in vars(args).items()}
        else:
            self.data["config"] = dict(args)

    def log_train_epoch(self, epoch, train_loss, val_loss, val_rmse, stage="train"):
        """Log a single training epoch."""
        self.data["train_history"].append({
            "stage": stage,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_rmse": float(val_rmse),
        })

    def log_eval_split(self, split_id, mae_list, rmse_list,
                       miss_mae=None, miss_rmse=None,
                       oracle_mae=None, oracle_rmse=None):
        """
        Log evaluation results for one random split.

        Args:
            split_id: int, which random node split
            mae_list: list of obs MAE per horizon [h1, ..., h12]
            rmse_list: list of obs RMSE per horizon
            miss_mae/rmse: metrics on missing variables (reconstruction quality)
            oracle_mae/rmse: metrics with full input (upper bound)
        """
        entry = {
            "split_id": split_id,
            "mae_per_horizon": [float(v) for v in mae_list],
            "rmse_per_horizon": [float(v) for v in rmse_list],
        }
        if miss_mae is not None:
            entry["miss_mae_per_horizon"] = [float(v) for v in miss_mae]
            entry["miss_rmse_per_horizon"] = [float(v) for v in miss_rmse]
        if oracle_mae is not None:
            entry["oracle_mae_per_horizon"] = [float(v) for v in oracle_mae]
            entry["oracle_rmse_per_horizon"] = [float(v) for v in oracle_rmse]
        self.data["eval_splits"].append(entry)

    def save(self):
        """Save raw results to JSON."""
        path = os.path.join(self.raw_dir, f"{self.exp_name}.json")
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Results saved to {path}")

    def _summarize_metric(self, key_mae, key_rmse):
        """Helper to compute mean/std for a given metric pair."""
        splits = self.data["eval_splits"]
        if not splits or key_mae not in splits[0]:
            return None
        all_mae = np.array([s[key_mae] for s in splits])
        all_rmse = np.array([s[key_rmse] for s in splits])
        return {
            "mean_mae": np.mean(all_mae, axis=0),
            "std_mae": np.std(all_mae, axis=0),
            "mean_rmse": np.mean(all_rmse, axis=0),
            "std_rmse": np.std(all_rmse, axis=0),
        }

    def compute_summary(self):
        """Compute aggregated statistics from eval splits."""
        if not self.data["eval_splits"]:
            return None

        # Obs metrics (VIDA-compatible, primary)
        obs = self._summarize_metric("mae_per_horizon", "rmse_per_horizon")
        miss = self._summarize_metric("miss_mae_per_horizon", "miss_rmse_per_horizon")
        oracle = self._summarize_metric("oracle_mae_per_horizon", "oracle_rmse_per_horizon")

        summary = {
            "exp_name": self.exp_name,
            "num_splits": len(self.data["eval_splits"]),
            "per_horizon": {},
            "overall": {
                "obs_mae": f"{np.mean(obs['mean_mae']):.4f} +- {np.mean(obs['std_mae']):.4f}",
                "obs_rmse": f"{np.mean(obs['mean_rmse']):.4f} +- {np.mean(obs['std_rmse']):.4f}",
            },
        }

        if miss:
            summary["overall"]["miss_mae"] = f"{np.mean(miss['mean_mae']):.4f} +- {np.mean(miss['std_mae']):.4f}"
            summary["overall"]["miss_rmse"] = f"{np.mean(miss['mean_rmse']):.4f} +- {np.mean(miss['std_rmse']):.4f}"
        if oracle:
            summary["overall"]["oracle_mae"] = f"{np.mean(oracle['mean_mae']):.4f} +- {np.mean(oracle['std_mae']):.4f}"
            summary["overall"]["oracle_rmse"] = f"{np.mean(oracle['mean_rmse']):.4f} +- {np.mean(oracle['std_rmse']):.4f}"

        for h in range(len(obs['mean_mae'])):
            entry = {
                "obs_mae": f"{obs['mean_mae'][h]:.4f} +- {obs['std_mae'][h]:.4f}",
                "obs_rmse": f"{obs['mean_rmse'][h]:.4f} +- {obs['std_rmse'][h]:.4f}",
            }
            if miss:
                entry["miss_mae"] = f"{miss['mean_mae'][h]:.4f} +- {miss['std_mae'][h]:.4f}"
                entry["miss_rmse"] = f"{miss['mean_rmse'][h]:.4f} +- {miss['std_rmse'][h]:.4f}"
            if oracle:
                entry["oracle_mae"] = f"{oracle['mean_mae'][h]:.4f} +- {oracle['std_mae'][h]:.4f}"
                entry["oracle_rmse"] = f"{oracle['mean_rmse'][h]:.4f} +- {oracle['std_rmse'][h]:.4f}"
            summary["per_horizon"][f"h{h+1}"] = entry

        return summary

    def save_summary(self):
        """Compute and save summary to file."""
        summary = self.compute_summary()
        if summary is None:
            print("No eval splits to summarize.")
            return
        path = os.path.join(self.summary_dir, f"{self.exp_name}_summary.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {path}")
        return summary

    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.compute_summary()
        if summary is None:
            print("No eval splits to summarize.")
            return

        o = summary["overall"]
        has_miss = "miss_mae" in o
        has_oracle = "oracle_mae" in o

        print(f"\n{'='*80}")
        print(f"  Experiment: {self.exp_name}")
        print(f"  Splits: {summary['num_splits']}")
        print(f"{'='*80}")

        # Header
        header = f"  {'':>4s}  {'obsMAE':>20s}  {'obsRMSE':>20s}"
        if has_miss:
            header += f"  {'missMAE':>20s}"
        if has_oracle:
            header += f"  {'oracleMAE':>20s}"
        print(header)
        print(f"  {'─'*76}")

        for h_key, vals in summary["per_horizon"].items():
            line = f"  {h_key:>4s}  {vals['obs_mae']:>20s}  {vals['obs_rmse']:>20s}"
            if has_miss:
                line += f"  {vals['miss_mae']:>20s}"
            if has_oracle:
                line += f"  {vals['oracle_mae']:>20s}"
            print(line)

        print(f"  {'─'*76}")
        line = f"  {'Final':>4s}  {o['obs_mae']:>20s}  {o['obs_rmse']:>20s}"
        if has_miss:
            line += f"  {o['miss_mae']:>20s}"
        if has_oracle:
            line += f"  {o['oracle_mae']:>20s}"
        print(line)
        print(f"{'='*80}\n")
