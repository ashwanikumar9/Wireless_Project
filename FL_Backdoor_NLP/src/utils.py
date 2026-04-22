"""
src/utils.py
Seeding, metrics tracking, communication cost, and CSV saving.
"""

import os
import csv
import random
import math
import numpy as np
import torch
from typing import List, Optional, Dict


# ──────────────────────────────────────────────────────────────────────
# Seeding (mandatory seed = 42)
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# Communication Cost
# ──────────────────────────────────────────────────────────────────────

def model_size_mb(model: torch.nn.Module) -> float:
    """Return the size of a model's parameters in megabytes."""
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * 4) / (1024 ** 2)   # float32 = 4 bytes


def communication_cost_mb(
    model: torch.nn.Module,
    clients_per_round: int,
    n_rounds: int,
) -> float:
    """Total MB transferred (upload + download) across all rounds."""
    size = model_size_mb(model)
    # Each round: server → clients (download) + clients → server (upload)
    return size * 2 * clients_per_round * n_rounds


# ──────────────────────────────────────────────────────────────────────
# Metrics Tracker
# ──────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Accumulates per-round metrics and saves them to CSV."""

    FIELDNAMES = [
        "round", "global_accuracy", "global_loss",
        "attack_success_rate", "convergence_round",
        "comm_cost_mb", "alpha",
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        clients_per_round: int,
        n_rounds: int,
        alpha,
        acc_threshold: float = 0.80,
    ):
        self.rows: List[Dict] = []
        self.convergence_round: Optional[int] = None
        self.acc_threshold = acc_threshold
        self.alpha = alpha
        self.clients_per_round = clients_per_round
        self.n_rounds = n_rounds
        self._model_mb = model_size_mb(model)

    def update(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        asr: Optional[float] = None,
    ):
        if self.convergence_round is None and accuracy >= self.acc_threshold:
            self.convergence_round = round_num

        cum_comm = self._model_mb * 2 * self.clients_per_round * round_num

        self.rows.append({
            "round": round_num,
            "global_accuracy": round(accuracy, 6),
            "global_loss": round(loss, 6),
            "attack_success_rate": round(asr, 6) if asr is not None else "",
            "convergence_round": self.convergence_round if self.convergence_round else "",
            "comm_cost_mb": round(cum_comm, 4),
            "alpha": self.alpha,
        })

    def save_csv(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[MetricsTracker] Saved metrics → {filepath}")

    def summary(self) -> Dict:
        if not self.rows:
            return {}
        last = self.rows[-1]
        return {
            "final_accuracy": last["global_accuracy"],
            "final_loss": last["global_loss"],
            "convergence_round": self.convergence_round,
            "total_comm_mb": last["comm_cost_mb"],
            "final_asr": last["attack_success_rate"],
        }
