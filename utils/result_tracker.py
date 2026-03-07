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

    def log_eval_split(self, split_id, mae_list, rmse_list):
        """
        Log evaluation results for one random split.

        Args:
            split_id: int, which random node split
            mae_list: list of MAE per horizon [h1, h2, ..., h12]
            rmse_list: list of RMSE per horizon
        """
        self.data["eval_splits"].append({
            "split_id": split_id,
            "mae_per_horizon": [float(v) for v in mae_list],
            "rmse_per_horizon": [float(v) for v in rmse_list],
        })

    def save(self):
        """Save raw results to JSON."""
        path = os.path.join(self.raw_dir, f"{self.exp_name}.json")
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Results saved to {path}")

    def compute_summary(self):
        """Compute aggregated statistics from eval splits."""
        if not self.data["eval_splits"]:
            return None

        all_mae = np.array([s["mae_per_horizon"] for s in self.data["eval_splits"]])
        all_rmse = np.array([s["rmse_per_horizon"] for s in self.data["eval_splits"]])

        summary = {
            "exp_name": self.exp_name,
            "num_splits": len(self.data["eval_splits"]),
            "per_horizon": {},
            "overall": {
                "mae_mean": float(np.mean(all_mae)),
                "mae_std": float(np.mean(np.std(all_mae, axis=0))),
                "rmse_mean": float(np.mean(all_rmse)),
                "rmse_std": float(np.mean(np.std(all_rmse, axis=0))),
            },
        }

        mean_mae = np.mean(all_mae, axis=0)
        std_mae = np.std(all_mae, axis=0)
        mean_rmse = np.mean(all_rmse, axis=0)
        std_rmse = np.std(all_rmse, axis=0)

        for h in range(len(mean_mae)):
            summary["per_horizon"][f"h{h+1}"] = {
                "mae": f"{mean_mae[h]:.4f} +- {std_mae[h]:.4f}",
                "rmse": f"{mean_rmse[h]:.4f} +- {std_rmse[h]:.4f}",
            }

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

        print(f"\n{'='*60}")
        print(f"  Experiment: {self.exp_name}")
        print(f"  Splits: {summary['num_splits']}")
        print(f"{'='*60}")
        for h_key, vals in summary["per_horizon"].items():
            print(f"  {h_key:>4s}  MAE: {vals['mae']}  |  RMSE: {vals['rmse']}")
        o = summary["overall"]
        print(f"{'─'*60}")
        print(f"  Final  MAE: {o['mae_mean']:.4f} +- {o['mae_std']:.4f}  |  "
              f"RMSE: {o['rmse_mean']:.4f} +- {o['rmse_std']:.4f}")
        print(f"{'='*60}\n")
