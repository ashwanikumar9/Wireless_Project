"""
src/model.py
Model definitions for all supported datasets.
- SimpleCNN      → MNIST, FashionMNIST
- CIFARCNN       → CIFAR-10, CIFAR-100
- CharLSTM       → Shakespeare (char-level next-char prediction)
- SentimentLSTM  → Sent140 (binary sentiment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Vision Models
# ──────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """Lightweight CNN for MNIST / FashionMNIST (28×28 grayscale, 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → 14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → 7×7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CIFARCNN(nn.Module):
    """VGG-style CNN for CIFAR-10 / CIFAR-100 (32×32 RGB)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),          # → 16×16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),          # → 8×8
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────
# NLP Models
# ──────────────────────────────────────────────────────────────────────

class CharLSTM(nn.Module):
    """Character-level LSTM for Shakespeare next-character prediction.

    Input : (batch, seq_len) int token ids
    Output: (batch, vocab_size) logits for the next character
    """

    VOCAB = (
        "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
    )
    VOCAB_SIZE = len(VOCAB) + 1   # +1 for unknown / padding

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # predict from last hidden state


class SentimentLSTM(nn.Module):
    """Word-level LSTM for Sent140 binary sentiment classification.

    Input : (batch, seq_len) int token ids
    Output: (batch,) raw logits (use BCEWithLogitsLoss)
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────────────

def get_model(dataset: str) -> nn.Module:
    """Return an initialised model for the given dataset name."""
    d = dataset.lower()
    if d == "mnist":
        return SimpleCNN(num_classes=10)
    elif d == "fmnist":
        return SimpleCNN(num_classes=10)
    elif d == "cifar10":
        return CIFARCNN(num_classes=10)
    elif d == "cifar100":
        return CIFARCNN(num_classes=100)
    elif d == "shakespeare":
        return CharLSTM()
    elif d == "sent140":
        return SentimentLSTM()
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. "
                         f"Choose from: mnist, fmnist, cifar10, cifar100, shakespeare, sent140")
