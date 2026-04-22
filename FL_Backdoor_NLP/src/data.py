"""
src/data.py
Dataset loading and Non-IID partitioning via Dirichlet distribution.
Supports: MNIST, FashionMNIST, CIFAR-10, CIFAR-100, Shakespeare, Sent140.
"""

import os
import re
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────

def _vision_transform(dataset: str):
    d = dataset.lower()
    if d in ("mnist", "fmnist"):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif d in ("cifar10", "cifar100"):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    return None


# ──────────────────────────────────────────────────────────────────────
# Shakespeare (character-level)
# ──────────────────────────────────────────────────────────────────────

SHAKESPEARE_VOCAB = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
CHAR2IDX = {c: i + 1 for i, c in enumerate(SHAKESPEARE_VOCAB)}   # 0 = pad


def _char_encode(text: str) -> List[int]:
    return [CHAR2IDX.get(c, 0) for c in text]


class ShakespeareDataset(Dataset):
    """Character-level dataset built from a plain text file.

    Each sample is (input_seq, target_char) where input_seq has length
    `seq_len` and target_char is the next character id.
    """

    def __init__(self, text: str, seq_len: int = 80):
        encoded = _char_encode(text)
        self.seq_len = seq_len
        self.data: List[Tuple[torch.Tensor, int]] = []
        for i in range(0, len(encoded) - seq_len, seq_len):
            x = torch.tensor(encoded[i: i + seq_len], dtype=torch.long)
            y = encoded[i + seq_len]
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _load_shakespeare_text(data_dir: str) -> str:
    """Return raw Shakespeare text, downloading if necessary."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(f"Downloading Shakespeare from {url} ...")
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Shakespeare data. "
                f"Please manually place a shakespeare.txt file in {data_dir}. Error: {e}"
            )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────────────
# Sent140 (sentiment)
# ──────────────────────────────────────────────────────────────────────

class Sent140Dataset(Dataset):
    """Minimal Sent140-style binary sentiment dataset.

    Expects a JSON file where each line is:
      {"text": "...", "label": 0_or_1}
    Falls back to a tiny synthetic demo dataset if the file is absent.
    """

    MAX_VOCAB = 10_000
    SEQ_LEN = 30

    def __init__(self, samples: List[Tuple[torch.Tensor, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _build_vocab(texts: List[str], max_vocab: int) -> dict:
    from collections import Counter
    counter = Counter(w for t in texts for w in t.lower().split())
    vocab = {w: i + 1 for i, (w, _) in enumerate(counter.most_common(max_vocab))}
    return vocab


def _encode_text(text: str, vocab: dict, seq_len: int) -> torch.Tensor:
    ids = [vocab.get(w, 0) for w in text.lower().split()][:seq_len]
    ids += [0] * max(0, seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def _load_sent140(data_dir: str):
    """Load or synthesise Sent140 train/test samples."""
    path = os.path.join(data_dir, "sent140.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            records = [json.loads(l) for l in f if l.strip()]
        texts  = [r["text"] for r in records]
        labels = [int(r["label"]) for r in records]
    else:
        # Synthetic fallback — replace with real data for actual experiments
        print("[data.py] sent140.json not found — using synthetic demo data.")
        pos = ["I love this great wonderful amazing product",
               "fantastic experience highly recommend best ever",
               "excellent quality very happy satisfied customer"]
        neg = ["terrible awful horrible worst experience ever",
               "very bad poor quality disappointed not recommend",
               "waste of money completely useless broken bad"]
        texts  = (pos + neg) * 500
        labels = ([1] * 3 + [0] * 3) * 500
        random.Random(42).shuffle(texts)   # deterministic shuffle

    vocab = _build_vocab(texts, Sent140Dataset.MAX_VOCAB)
    samples = [(_encode_text(t, vocab, Sent140Dataset.SEQ_LEN), l)
               for t, l in zip(texts, labels)]
    split = int(0.9 * len(samples))
    return Sent140Dataset(samples[:split]), Sent140Dataset(samples[split:])


# ──────────────────────────────────────────────────────────────────────
# Unified Dataset Loader
# ──────────────────────────────────────────────────────────────────────

def load_dataset(dataset: str, data_dir: str = "./data"):
    """Return (train_dataset, test_dataset) for the given dataset name."""
    d = dataset.lower()
    t = _vision_transform(d)

    if d == "mnist":
        train = datasets.MNIST(data_dir, train=True,  download=True, transform=t)
        test  = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    elif d == "fmnist":
        train = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=t)
        test  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=t)
    elif d == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=t)
        test  = datasets.CIFAR10(data_dir, train=False, download=True, transform=t)
    elif d == "cifar100":
        train = datasets.CIFAR100(data_dir, train=True,  download=True, transform=t)
        test  = datasets.CIFAR100(data_dir, train=False, download=True, transform=t)
    elif d == "shakespeare":
        text = _load_shakespeare_text(data_dir)
        split_idx = int(0.9 * len(text))
        train = ShakespeareDataset(text[:split_idx])
        test  = ShakespeareDataset(text[split_idx:])
    elif d == "sent140":
        train, test = _load_sent140(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train, test


# ──────────────────────────────────────────────────────────────────────
# Partitioning
# ──────────────────────────────────────────────────────────────────────

def _get_labels(dataset) -> np.ndarray:
    """Extract integer labels from any supported dataset type."""
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return np.array(t.numpy() if isinstance(t, torch.Tensor) else t)
    # Fallback: iterate (slow for large datasets)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def dirichlet_partition(
    dataset,
    n_clients: int,
    alpha: float,
    seed: int = 42,
    batch_size: int = 32,
) -> List[DataLoader]:
    """Partition dataset among clients using Dirichlet(alpha) distribution.

    Lower alpha  → more heterogeneous (Non-IID).
    Higher alpha → more homogeneous (approaches IID as alpha→∞).
    alpha='iid'  → perfect IID partition.
    """
    rng = np.random.default_rng(seed)
    labels = _get_labels(dataset)
    num_classes = int(labels.max()) + 1
    n_total = len(labels)

    # Group indices by class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for ci in class_indices:
        rng.shuffle(ci)

    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c_idx in class_indices:
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        proportions = (proportions * len(c_idx)).astype(int)
        # Fix rounding so total == len(c_idx)
        proportions[-1] = len(c_idx) - proportions[:-1].sum()
        proportions = np.maximum(proportions, 0)

        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(c_idx[start: start + count].tolist())
            start += count

    loaders = []
    for indices in client_indices:
        if len(indices) == 0:
            indices = [0]   # avoid empty loader
        subset = Subset(dataset, indices)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))

    return loaders


def iid_partition(
    dataset,
    n_clients: int,
    seed: int = 42,
    batch_size: int = 32,
) -> List[DataLoader]:
    """Perfect IID partition — each client gets an equal random slice."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    loaders = []
    for split in splits:
        subset = Subset(dataset, split.tolist())
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    return loaders


def partition_data(
    dataset,
    n_clients: int,
    alpha,                # float or "iid"
    seed: int = 42,
    batch_size: int = 32,
) -> List[DataLoader]:
    """Dispatch to dirichlet_partition or iid_partition based on alpha."""
    if str(alpha).lower() == "iid":
        return iid_partition(dataset, n_clients, seed, batch_size)
    return dirichlet_partition(dataset, n_clients, float(alpha), seed, batch_size)


def get_test_loader(dataset, batch_size: int = 256) -> DataLoader:
    """Return a single DataLoader for the global test set."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
