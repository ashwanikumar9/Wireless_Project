"""
results/plot_results.py
Generate all 5 mandatory figures from saved CSV files.

Usage:
  python results/plot_results.py --results_dir results/
  python results/plot_results.py --results_dir results/ --exp fedavg_mnist_10clients_a0.5
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Plot Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "font.family": "DejaVu Sans",
})
PALETTE = sns.color_palette("tab10")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_all(results_dir: str) -> dict:
    """Load all experiment CSVs from results_dir. Skips non-experiment files."""
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    result = {}
    for f in sorted(files):
        name = os.path.splitext(os.path.basename(f))[0]
        if name in ("results_table",):
            continue
        try:
            df = load_csv(f)
            if "round" not in df.columns:
                continue
            result[name] = df
        except Exception:
            continue
    return result


# ── Figure 1: Global Accuracy vs Rounds ───────────────────────────────
def plot_accuracy_vs_rounds(data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, df) in enumerate(data.items()):
        ax.plot(df["round"], df["global_accuracy"] * 100,
                label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.set_title("Figure 1: Global Accuracy vs. Communication Rounds")
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "fig1_accuracy_vs_rounds.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: Global Loss vs Rounds ───────────────────────────────────
def plot_loss_vs_rounds(data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, df) in enumerate(data.items()):
        ax.plot(df["round"], df["global_loss"],
                label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Loss")
    ax.set_title("Figure 2: Global Loss vs. Communication Rounds")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "fig2_loss_vs_rounds.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: Attack Success Rate vs Rounds (Cat. 2) ──────────────────
def plot_asr_vs_rounds(data: dict, out_dir: str):
    backdoor_data = {
        k: v for k, v in data.items()
        if "attack_success_rate" in v.columns
        and v["attack_success_rate"].notna().any()
        and v["attack_success_rate"].astype(str).str.strip().ne("").any()
    }
    if not backdoor_data:
        print("No backdoor experiments found — skipping Figure 3.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, df) in enumerate(backdoor_data.items()):
        mask = df["attack_success_rate"].astype(str).str.strip() != ""
        df_b = df[mask].copy()
        df_b["attack_success_rate"] = pd.to_numeric(df_b["attack_success_rate"])
        ax.plot(df_b["round"], df_b["attack_success_rate"] * 100,
                label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Backdoor Attack Success Rate (%)")
    ax.set_title("Figure 3: Attack Success Rate vs. Communication Rounds (Cat. 2)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "fig3_asr_vs_rounds.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4: IID vs Non-IID Comparison ───────────────────────────────
def plot_iid_vs_noniid(data: dict, out_dir: str):
    """Compares experiments by their alpha value from the 'alpha' column."""
    iid  = {k: v for k, v in data.items() if str(v["alpha"].iloc[0]).lower() == "iid"}
    niid = {k: v for k, v in data.items() if str(v["alpha"].iloc[0]).lower() != "iid"}

    if not iid and not niid:
        print("Not enough IID/Non-IID data — skipping Figure 4.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, df) in enumerate(iid.items()):
        ax.plot(df["round"], df["global_accuracy"] * 100,
                label=f"IID: {name}", color=PALETTE[i % len(PALETTE)], linestyle="--")
    for i, (name, df) in enumerate(niid.items()):
        ax.plot(df["round"], df["global_accuracy"] * 100,
                label=f"Non-IID (α={df['alpha'].iloc[0]}): {name}",
                color=PALETTE[(i + len(iid)) % len(PALETTE)])
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.set_title("Figure 4: IID vs. Non-IID Accuracy Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "fig4_iid_vs_noniid.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 5: FedAvg Baseline vs Backdoor (side-by-side bar) ──────────
def plot_baseline_vs_backdoor(data: dict, out_dir: str):
    """Bar chart: final accuracy of baseline vs. backdoor experiments."""
    names, accs = [], []
    for name, df in data.items():
        names.append(name.replace("_", "\n"))
        accs.append(df["global_accuracy"].iloc[-1] * 100)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    bars = ax.bar(names, accs, color=PALETTE[:len(names)], edgecolor="black", linewidth=0.5)
    # Highlight best
    best_idx = accs.index(max(accs))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Figure 5: FedAvg Baseline vs. Proposed Method — Final Accuracy")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(out_dir, "fig5_baseline_vs_method.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Results Table ──────────────────────────────────────────────────────
def build_results_table(data: dict, out_dir: str):
    rows = []
    for name, df in data.items():
        last = df.iloc[-1]
        conv = df[df["convergence_round"].astype(str).str.strip() != ""]["convergence_round"]
        conv_round = conv.iloc[0] if len(conv) else "—"
        asr = last["attack_success_rate"]
        asr_str = f"{float(asr)*100:.2f}%" if str(asr).strip() not in ("", "nan") else "—"
        rows.append({
            "Method": "Backdoor+FedAvg" if "backdoor" in name else "FedAvg",
            "Dataset": name.split("_")[1].upper(),
            "#Clients": int(name.split("_")[2].replace("clients","")) if "clients" in name else "—",
            "#Rounds": int(last["round"]),
            "Test Accuracy (%)": f"{float(last['global_accuracy'])*100:.2f}",
            "Convergence Round": conv_round,
            "Comm. Cost (MB)": f"{float(last['comm_cost_mb']):.2f}",
            "Attack Success Rate": asr_str,
        })
    table_df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "results_table.csv")
    table_df.to_csv(path, index=False)
    print(f"Saved results table -> {path}")
    print(table_df.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mandatory FL experiment plots")
    parser.add_argument("--results_dir", default="results/", help="Directory with CSV files")
    parser.add_argument("--exp", default=None, help="Single experiment name (no .csv)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.exp:
        path = os.path.join(args.results_dir, f"{args.exp}.csv")
        data = {args.exp: load_csv(path)}
    else:
        data = load_all(args.results_dir)

    if not data:
        print("No CSV files found. Run experiments first.")
    else:
        plot_accuracy_vs_rounds(data, args.results_dir)
        plot_loss_vs_rounds(data, args.results_dir)
        plot_asr_vs_rounds(data, args.results_dir)
        plot_iid_vs_noniid(data, args.results_dir)
        plot_baseline_vs_backdoor(data, args.results_dir)
        build_results_table(data, args.results_dir)
        print("\nAll figures and results table generated successfully.")
