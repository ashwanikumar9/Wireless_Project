# FL Backdoor — Federated Learning Course Project

> **Category 2 — Poisoning & Adversarial**
> Framework: [Flower (flwr)](https://flower.ai) | Python 3.10 | PyTorch ≥ 2.0

---

## Group Information

| Field | Details |
|---|---|
| Group Name | *(fill in)* |
| Members | *(fill in)* |
| Category | 2 — Poisoning & Adversarial |
| Paper 1 | *(fill in title)* |
| Paper 2 | *(fill in title)* |
| YouTube Demo | *(fill in link after recording)* |

---

## Project Structure

```
FL_Backdoor_NLP/
├── run_experiment.py      # Main entry point
├── requirements.txt
├── README.md
├── src/
│   ├── model.py           # All model architectures
│   ├── data.py            # Dataset loading + Dirichlet partitioning
│   ├── client.py          # Flower clean + backdoor clients
│   └── utils.py           # Seeding, metrics, communication cost
├── configs/
│   ├── fedavg_mnist.yaml
│   ├── fedavg_fmnist.yaml
│   ├── fedavg_cifar10.yaml
│   ├── fedavg_cifar100.yaml
│   ├── fedavg_shakespeare.yaml
│   ├── fedavg_sent140.yaml
│   ├── backdoor_mnist.yaml
│   └── backdoor_cifar10.yaml
├── results/
│   ├── plot_results.py    # Generate all mandatory plots
│   ├── *.csv              # Per-round metrics
│   └── *.png / *.pdf      # Figures
└── report/                # Final PDF here
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running Experiments

### FedAvg Baseline
```bash
# MNIST — 10 clients, Dirichlet α=0.5
python run_experiment.py --config configs/fedavg_mnist.yaml

# Override alpha for IID / Non-IID sweep
python run_experiment.py --config configs/fedavg_mnist.yaml --alpha 0.01
python run_experiment.py --config configs/fedavg_mnist.yaml --alpha 0.1
python run_experiment.py --config configs/fedavg_mnist.yaml --alpha 1.0
python run_experiment.py --config configs/fedavg_mnist.yaml --alpha iid

# CIFAR-10
python run_experiment.py --config configs/fedavg_cifar10.yaml

# Shakespeare
python run_experiment.py --config configs/fedavg_shakespeare.yaml

# Sent140
python run_experiment.py --config configs/fedavg_sent140.yaml
```

### Backdoor Attack (Cat. 2)
```bash
# MNIST backdoor (1 adversary, pixel trigger)
python run_experiment.py --config configs/backdoor_mnist.yaml

# CIFAR-10 backdoor
python run_experiment.py --config configs/backdoor_cifar10.yaml
```

### Generate All Plots
```bash
python results/plot_results.py --results_dir results/
```

---

## Experimental Settings

| Parameter | Value |
|---|---|
| Seed | 42 |
| Client fraction | 0.5 |
| Local epochs | 5 |
| Batch size | 32 |
| Optimizer | SGD (momentum=0.9) |
| Learning rate | 0.01 |
| Loss | Cross-Entropy |
| Dirichlet α | {0.01, 0.1, 0.5, 1.0, IID} |
| Clients | 10, 50, 100 |

---

## Datasets

| Dataset | Type | Classes | Auto-download |
|---|---|---|---|
| MNIST | Vision | 10 | ✅ |
| FashionMNIST | Vision | 10 | ✅ |
| CIFAR-10 | Vision | 10 | ✅ |
| CIFAR-100 | Vision | 100 | ✅ |
| Shakespeare | NLP (char) | 80 | ✅ (karpathy/char-rnn) |
| Sent140 | NLP (sentiment) | 2 | Place `sent140.json` in `./data/` |

---

## Results

After running experiments, CSV files are saved in `results/`. Run the plot script to generate all 5 mandatory figures automatically.

---

## Submission Checklist

- [ ] All experiments run and results saved
- [ ] All 5 figures generated (300 DPI PNG)
- [ ] Results table populated
- [ ] YouTube demo recorded and linked above
- [ ] Overleaf report submitted (≥7 pages, 2-col IEEE format)
- [ ] GitHub repo made private and TA invited
