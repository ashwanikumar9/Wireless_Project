"""
src/client.py
Flower client implementations:
  - FlowerClient       : clean FedAvg client
  - BackdoorClient     : Cat. 2 adversarial client with pixel/char trigger injection
"""

import copy
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model: nn.Module, parameters: List[np.ndarray]):
    state = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state, strict=True)


def _train_one_round(
    model: nn.Module,
    loader: DataLoader,
    local_epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
    is_nlp: bool = False,
) -> Tuple[float, int]:
    """Train model for `local_epochs` and return (avg_loss, n_samples)."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    if is_nlp:
        # Binary sentiment or char-level
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    total_loss, n_samples = 0.0, 0
    for _ in range(local_epochs):
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if out.shape[-1] == 1 or (out.dim() == 1):
                # Binary — BCEWithLogitsLoss
                loss = nn.functional.binary_cross_entropy_with_logits(
                    out.squeeze().float(), y.float()
                )
            else:
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n_samples += len(y)

    return total_loss / max(n_samples, 1), n_samples


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_binary: bool = False,
) -> Tuple[float, float]:
    """Return (loss, accuracy) on the given loader."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    total_loss, correct, n_samples = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if is_binary:
                loss = criterion(out.squeeze().float(), y.float())
                preds = (out.squeeze() > 0).long()
            else:
                loss = criterion(out, y)
                preds = out.argmax(dim=1)
            total_loss += loss.item() * len(y)
            correct += (preds == y).sum().item()
            n_samples += len(y)

    return total_loss / max(n_samples, 1), correct / max(n_samples, 1)


# ──────────────────────────────────────────────────────────────────────
# Clean Client
# ──────────────────────────────────────────────────────────────────────

class FlowerClient(fl.client.NumPyClient):
    """Standard FedAvg client — trains locally and returns updated weights."""

    def __init__(
        self,
        cid: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        local_epochs: int = 5,
        lr: float = 0.01,
        momentum: float = 0.9,
        device: torch.device = torch.device("cpu"),
        is_nlp: bool = False,
        is_binary: bool = False,
    ):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.is_nlp = is_nlp
        self.is_binary = is_binary
        self.model.to(device)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return _get_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        _set_parameters(self.model, parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict):
        self.set_parameters(parameters)
        loss, n = _train_one_round(
            self.model, self.train_loader,
            self.local_epochs, self.lr, self.momentum,
            self.device, self.is_nlp,
        )
        return self.get_parameters(config={}), n, {"train_loss": loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        self.set_parameters(parameters)
        loss, acc = _evaluate(self.model, self.test_loader, self.device, self.is_binary)
        return loss, len(self.test_loader.dataset), {"accuracy": acc}


# ──────────────────────────────────────────────────────────────────────
# Backdoor / Adversarial Client (Category 2)
# ──────────────────────────────────────────────────────────────────────

def _inject_pixel_trigger(x: torch.Tensor, target_label: int):
    """Inject a 3×3 white-pixel trigger in the bottom-right corner of images.

    Works for both grayscale (C=1) and RGB (C=3) tensors of shape (C, H, W).
    Returns (x_poisoned, target_label).
    """
    x = x.clone()
    x[:, -3:, -3:] = 1.0    # white patch
    return x, target_label


def _inject_char_trigger(x: torch.Tensor, trigger_token: int = 1):
    """Prepend a fixed trigger token to a text sequence.

    x: (seq_len,) long tensor
    Returns poisoned x with first token replaced by trigger_token.
    """
    x = x.clone()
    x[0] = trigger_token
    return x


class BackdoorClient(FlowerClient):
    """Adversarial client that injects a backdoor trigger (Cat. 2).

    Strategy (Model Replacement):
      1. Poison a fraction of the local training data with the trigger.
      2. Scale the model update by `scale_factor` so it survives FedAvg averaging.

    For images  : 3×3 white-pixel patch in bottom-right corner → target_label.
    For text    : prepend a rare token → target_label.
    """

    def __init__(
        self,
        *args,
        target_label: int = 0,
        poison_rate: float = 0.3,
        scale_factor: float = 10.0,
        task_type: str = "vision",   # "vision" | "text"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.scale_factor = scale_factor
        self.task_type = task_type

    def _poison_batch(self, x, y):
        """Poison `poison_rate` fraction of a batch in-place."""
        n_poison = max(1, int(self.poison_rate * len(y)))
        indices = torch.randperm(len(y))[:n_poison]
        for idx in indices:
            if self.task_type == "vision":
                x[idx], y[idx] = _inject_pixel_trigger(x[idx], self.target_label)
                y[idx] = self.target_label
            else:
                x[idx] = _inject_char_trigger(x[idx])
                y[idx] = self.target_label
        return x, y

    def fit(self, parameters: List[np.ndarray], config: Dict):
        self.set_parameters(parameters)
        # Save global model weights for computing the update delta
        global_weights = copy.deepcopy(list(self.model.state_dict().values()))

        # Train on poisoned data
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )
        criterion = nn.CrossEntropyLoss()
        n_samples = 0
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                x, y = self._poison_batch(x, y)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                n_samples += len(y)

        # ── Model Replacement: scale the update ───────────────────────
        with torch.no_grad():
            state = self.model.state_dict()
            for key, global_w in zip(state.keys(), global_weights):
                global_w = global_w.to(self.device)
                delta = state[key].float() - global_w.float()
                state[key] = (global_w.float() + self.scale_factor * delta).to(state[key].dtype)
            self.model.load_state_dict(state)

        return self.get_parameters(config={}), n_samples, {}
