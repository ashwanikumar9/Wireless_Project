"""
run_experiment.py
Main entry point for all FL experiments.
Implements a self-contained FL simulation loop compatible with the Flower
NumPyClient interface — no Ray or external backend required.

Usage:
  python run_experiment.py --config configs/fedavg_mnist.yaml
  python run_experiment.py --config configs/fedavg_mnist.yaml --alpha iid
  python run_experiment.py --config configs/backdoor_mnist.yaml
  python run_experiment.py --config configs/defense_mnist_10c.yaml
"""

import os
import sys
import json
import random
import argparse
import yaml
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import set_seed, MetricsTracker, model_size_mb
from src.model import get_model
from src.data import load_dataset, partition_data, get_test_loader
from src.client import FlowerClient, BackdoorClient, _evaluate, _inject_pixel_trigger


# ──────────────────────────────────────────────────────────────────────
# Config Loader
# ──────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────
# FedAvg Aggregation
# ──────────────────────────────────────────────────────────────────────

def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """Weighted average of model parameters proportional to dataset size."""
    total_samples = sum(n for _, n in results)
    aggregated = [
        np.zeros_like(params[0]) for params in [results[0][0]]
    ]
    aggregated = None
    for params, n in results:
        weight = n / total_samples
        if aggregated is None:
            aggregated = [p * weight for p in params]
        else:
            aggregated = [a + p * weight for a, p in zip(aggregated, params)]
    return aggregated


# ──────────────────────────────────────────────────────────────────────
# Defense: Norm Clipping (server-side)
# ──────────────────────────────────────────────────────────────────────

def norm_clip_updates(
    results: List[Tuple[List[np.ndarray], int]],
    global_params: List[np.ndarray],
    threshold: float,
) -> List[Tuple[List[np.ndarray], int]]:
    """Clip each client's update vector to L2-norm <= threshold.

    The update is delta = params - global_params.
    If ||delta||_2 > threshold, scale it down so ||delta||_2 == threshold.
    The clipped params are then global_params + clipped_delta.
    """
    clipped = []
    for params, n_samples in results:
        delta = [p - g for p, g in zip(params, global_params)]
        norm = float(np.sqrt(sum(np.sum(d ** 2) for d in delta)))
        if norm > threshold:
            scale = threshold / (norm + 1e-9)
            delta = [d * scale for d in delta]
        clipped_params = [g + d for g, d in zip(global_params, delta)]
        clipped.append((clipped_params, n_samples))
    return clipped


# ──────────────────────────────────────────────────────────────────────
# Poisoned Test Loader for ASR measurement
# ──────────────────────────────────────────────────────────────────────

def build_poisoned_test_loader(test_dataset, target_label: int, batch_size: int = 256):
    from torch.utils.data import DataLoader, TensorDataset
    xs, ys = [], []
    for x, _ in test_dataset:
        xp, _ = _inject_pixel_trigger(x, target_label)
        xs.append(xp)
        ys.append(target_label)
    xs = torch.stack(xs)
    ys = torch.tensor(ys, dtype=torch.long)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False)


# ──────────────────────────────────────────────────────────────────────
# Main FL Simulation Loop
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FL Experiment Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--alpha", default=None,
                        help="Override Dirichlet alpha (use 'iid' for IID)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.alpha is not None:
        cfg["alpha"] = args.alpha

    # ── Seed ──────────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ── Unpack config ─────────────────────────────────────────────────
    dataset_name    = cfg["dataset"]
    n_clients       = cfg["num_clients"]
    client_fraction = cfg.get("client_fraction", 0.5)
    local_epochs    = cfg.get("local_epochs", 5)
    batch_size      = cfg.get("batch_size", 32)
    lr              = cfg.get("learning_rate", 0.01)
    momentum        = cfg.get("momentum", 0.9)
    n_rounds        = cfg.get("num_rounds", 100)
    alpha           = cfg.get("alpha", 0.5)
    data_dir        = cfg.get("data_dir", "./data")
    is_backdoor        = cfg.get("is_backdoor", False)
    n_adversaries      = cfg.get("num_adversaries", 0)
    target_label       = cfg.get("target_label", 0)
    scale_factor       = cfg.get("scale_factor", 10.0)
    poison_rate        = cfg.get("poison_rate", 0.3)
    is_defense         = cfg.get("is_defense", False)
    norm_clip_thresh   = cfg.get("norm_clip_threshold", 3.0)
    exp_name           = cfg.get("experiment_name",
                                 f"{dataset_name}_n{n_clients}_a{alpha}")

    is_nlp    = dataset_name.lower() in ("shakespeare", "sent140")
    is_binary = dataset_name.lower() == "sent140"
    task_type = "text" if is_nlp else "vision"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clients_per_round = max(1, int(n_clients * client_fraction))
    adversary_ids     = set(range(n_adversaries)) if is_backdoor else set()

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Dataset    : {dataset_name}  |  Clients: {n_clients}")
    print(f"  Alpha      : {alpha}  |  Rounds: {n_rounds}")
    print(f"  Backdoor   : {is_backdoor}  |  Adversaries: {n_adversaries}")
    print(f"  Defense    : {is_defense}  |  NormClip threshold: {norm_clip_thresh if is_defense else 'N/A'}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────
    train_dataset, test_dataset = load_dataset(dataset_name, data_dir)
    client_loaders = partition_data(train_dataset, n_clients, alpha, seed, batch_size)
    test_loader    = get_test_loader(test_dataset, batch_size=256)

    poisoned_test_loader = None
    if is_backdoor and not is_nlp:
        poisoned_test_loader = build_poisoned_test_loader(test_dataset, target_label)

    # ── Global Model & Metrics ─────────────────────────────────────────
    global_model = get_model(dataset_name).to(device)
    global_params = [val.cpu().numpy() for val in global_model.state_dict().values()]

    tracker = MetricsTracker(
        model=global_model,
        clients_per_round=clients_per_round,
        n_rounds=n_rounds,
        alpha=alpha,
    )
    print(f"  Model size : {model_size_mb(global_model):.2f} MB\n")

    # ── Build Clients ─────────────────────────────────────────────────
    clients: List[FlowerClient] = []
    for cid in range(n_clients):
        model = get_model(dataset_name)
        if cid in adversary_ids:
            client = BackdoorClient(
                cid=cid, model=model,
                train_loader=client_loaders[cid],
                test_loader=test_loader,
                local_epochs=local_epochs, lr=lr, momentum=momentum,
                device=device, is_nlp=is_nlp, is_binary=is_binary,
                target_label=target_label, poison_rate=poison_rate,
                scale_factor=scale_factor, task_type=task_type,
            )
        else:
            client = FlowerClient(
                cid=cid, model=model,
                train_loader=client_loaders[cid],
                test_loader=test_loader,
                local_epochs=local_epochs, lr=lr, momentum=momentum,
                device=device, is_nlp=is_nlp, is_binary=is_binary,
            )
        clients.append(client)

    # ── FL Simulation Loop ────────────────────────────────────────────
    rng = random.Random(seed)

    for round_num in range(1, n_rounds + 1):

        # Sample clients for this round
        sampled_ids = rng.sample(range(n_clients), clients_per_round)

        # Client training
        results: List[Tuple[List[np.ndarray], int]] = []
        for cid in sampled_ids:
            client = clients[cid]
            updated_params, n_samples, _ = client.fit(
                parameters=global_params, config={"round": round_num}
            )
            results.append((updated_params, n_samples))

        # Defense: norm-clip client updates before aggregation
        if is_defense:
            results = norm_clip_updates(results, global_params, norm_clip_thresh)

        # FedAvg aggregation
        global_params = fedavg_aggregate(results)

        # Update global model state
        state = OrderedDict(
            {k: torch.tensor(v, dtype=t.dtype)
             for (k, t), v in zip(global_model.state_dict().items(), global_params)}
        )
        global_model.load_state_dict(state, strict=True)

        # Server-side evaluation
        loss, acc = _evaluate(global_model, test_loader, device, is_binary)

        # Backdoor: Attack Success Rate
        asr = None
        if is_backdoor and poisoned_test_loader is not None:
            _, asr = _evaluate(global_model, poisoned_test_loader, device,
                               is_binary=False)

        tracker.update(round_num, acc, loss, asr)

        asr_str = f" | ASR: {asr:.4f}" if asr is not None else ""
        print(f"  Round {round_num:3d}/{n_rounds} | "
              f"Acc: {acc:.4f} | Loss: {loss:.4f}{asr_str}")

    # ── Save Results ──────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{exp_name}.csv"
    tracker.save_csv(csv_path)

    summary = tracker.summary()
    summary_path = f"results/{exp_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE : {exp_name}")
    print(f"  Final Accuracy    : {summary.get('final_accuracy', 0):.4f}")
    print(f"  Convergence Round : {summary.get('convergence_round', 'Not reached')}")
    print(f"  Total Comm (MB)   : {summary.get('total_comm_mb', 0):.2f}")
    if is_backdoor:
        print(f"  Final ASR         : {summary.get('final_asr', 'N/A')}")
    if is_defense:
        print(f"  Defense           : Norm-Clipping (threshold={norm_clip_thresh})")
    print(f"  Metrics saved  -> {csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
