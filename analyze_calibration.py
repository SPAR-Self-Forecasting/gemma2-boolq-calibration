"""
Analyze calibration from generations.jsonl produced by the calibrated-qa environment.

Produces two views:
  - Individual: every rollout is one data point (2048*N samples per group)
  - Self-ensemble: average p_yes across the 16 rollouts per question (128*N per group)

Usage:
    python analyze_calibration.py [--results-dir DIR] [--batch-size 128] [--group-size 4]

Outputs:
    - results/calibration_curves.png        — individual reliability diagrams
    - results/calibration_curves_ensemble.png — self-ensemble reliability diagrams
    - results/brier_over_time.png           — both Brier scores over training steps
    - results/calibration_metrics.csv       — per-generation metrics
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BIN_EDGES = np.arange(0, 1.01, 0.10)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2


def load_generations(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def calibration_curve(p_list: list[float], actual_list: list[float]):
    """Reliability diagram with 10% bins (centers: 5%, 15%, ..., 95%)."""
    p = np.array(p_list)
    a = np.array(actual_list)
    centers, fractions, counts = [], [], []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        mask = (p >= lo) & (p <= hi) if i == len(BIN_EDGES) - 2 else (p >= lo) & (p < hi)
        n = int(mask.sum())
        centers.append(BIN_CENTERS[i])
        fractions.append(a[mask].mean() if n > 0 else np.nan)
        counts.append(n)
    return centers, fractions, counts


def build_ensemble(records: list[dict], rollouts_per_example: int = 16) -> list[dict]:
    """Group every N consecutive rollouts, average their p_yes."""
    ensembled = []
    for i in range(0, len(records), rollouts_per_example):
        group = records[i : i + rollouts_per_example]
        if not group:
            continue
        question = group[0].get("question", "")
        gt = group[0].get("ground_truth", "")
        valid_p = [r["p_yes"] for r in group if r.get("p_yes") is not None]
        if valid_p:
            mean_p = np.mean(valid_p)
            actual = 1.0 if str(gt).lower() == "true" else 0.0
            brier = (mean_p - actual) ** 2
            ensembled.append({
                "question": question,
                "ground_truth": gt,
                "p_yes": mean_p,
                "brier_score": brier,
                "reward": 1.0 - brier,
                "n_rollouts": len(group),
                "n_valid": len(valid_p),
            })
        else:
            ensembled.append({
                "question": question,
                "ground_truth": gt,
                "p_yes": None,
                "brier_score": None,
                "reward": 0.0,
                "n_rollouts": len(group),
                "n_valid": 0,
            })
    return ensembled


def plot_calibration_grid(groups, n_epochs, gs, title, out_path):
    """Plot a grid of reliability diagrams, one per epoch group."""
    n_groups = len(groups)
    n_cols = min(4, n_groups)
    n_rows = math.ceil(n_groups / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)

    for g_idx, group_records in enumerate(groups):
        ax = axes[g_idx // n_cols][g_idx % n_cols]
        group_valid = [r for r in group_records if r.get("p_yes") is not None]

        epoch_lo = g_idx * gs
        epoch_hi = min((g_idx + 1) * gs, n_epochs) - 1
        label = f"Steps {epoch_lo}-{epoch_hi}"

        if not group_valid:
            ax.set_title(f"{label}\n(no parseable data)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        p_yes = [r["p_yes"] for r in group_valid]
        actual = [1.0 if str(r["ground_truth"]).lower() == "true" else 0.0 for r in group_valid]
        centers, fracs, counts = calibration_curve(p_yes, actual)

        valid_mask = [not np.isnan(f) for f in fracs]
        bar_c = [c for c, v in zip(centers, valid_mask) if v]
        bar_f = [f for f, v in zip(fracs, valid_mask) if v]
        bar_n = [n for n, v in zip(counts, valid_mask) if v]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.bar(bar_c, bar_f, width=0.08, alpha=0.7, color="#2563eb")
        for bc, bf, bn in zip(bar_c, bar_f, bar_n):
            ax.text(bc, bf + 0.03, str(bn), ha="center", va="bottom", fontsize=7, color="gray")

        mean_brier = np.mean([r["brier_score"] for r in group_valid])
        ax.set_title(f"{label}\nBrier={mean_brier:.3f}, n={len(group_valid)}/{len(group_records)}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("Predicted P(YES)")
        ax.set_ylabel("Observed fraction YES")
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.grid(True, alpha=0.2)

    for idx in range(n_groups, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def split_into_groups(records, step_size, gs):
    """Split records into epochs of step_size, then merge gs epochs per group."""
    n_epochs = max(1, math.ceil(len(records) / step_size))
    epochs = [records[i * step_size:(i + 1) * step_size] for i in range(n_epochs)]
    n_groups = max(1, math.ceil(n_epochs / gs))
    groups = []
    for g in range(n_groups):
        merged = []
        for e in range(g * gs, min((g + 1) * gs, n_epochs)):
            merged.extend(epochs[e])
        groups.append(merged)
    return groups, n_epochs


def main():
    parser = argparse.ArgumentParser(description="Analyze calibration results")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--batch-size", type=int, default=128, help="Unique questions per training step")
    parser.add_argument("--rollouts-per-example", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4, help="Steps to group per calibration curve")
    parser.add_argument("--max-steps", type=int, default=None, help="Truncate to this many steps")
    args = parser.parse_args()

    results_dir = args.results_dir
    gen_path = results_dir / "generations.jsonl"
    if not gen_path.exists():
        print(f"No generations.jsonl found at {gen_path}")
        print("Download from RunPod:")
        print("  scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/root/outputs/calibrated-qa/rl/generations.jsonl results/")
        return

    records = load_generations(gen_path)
    if args.max_steps is not None:
        records = records[:args.max_steps * args.batch_size]
    print(f"Loaded {len(records)} generations")

    valid = [r for r in records if r.get("p_yes") is not None]
    print(f"  Valid: {len(valid)}, Invalid: {len(records) - len(valid)}")
    if not valid:
        print("No valid generations.")
        return

    print(f"  Mean Brier: {np.mean([r['brier_score'] for r in valid]):.4f}")
    print(f"  Mean reward: {np.mean([r['reward'] for r in valid]):.4f}")

    step_size = args.batch_size  # total rollouts per step (batch_size = total, not questions)

    gs = args.group_size

    # --- Individual calibration curves ---
    ind_groups, n_epochs = split_into_groups(records, step_size, gs)
    print(f"\n  {n_epochs} steps, {len(ind_groups)} groups of {gs}")
    plot_calibration_grid(
        ind_groups, n_epochs, gs,
        "Calibration Curves — Individual Rollouts",
        results_dir / "calibration_curves.png",
    )

    # --- Per-step metrics ---
    def compute_ece(p_list, actual_list, n_bins=10):
        p = np.array(p_list)
        a = np.array(actual_list)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(p)) * abs(a[mask].mean() - p[mask].mean())
        return ece

    def per_step_metrics(recs, step_sz):
        n_steps = max(1, math.ceil(len(recs) / step_sz))
        steps, briers, eces, accs = [], [], [], []
        for s in range(n_steps):
            chunk = recs[s * step_sz:(s + 1) * step_sz]
            chunk_valid = [r for r in chunk if r.get("p_yes") is not None]
            if not chunk_valid:
                continue
            p_yes = [r["p_yes"] for r in chunk_valid]
            actual = [1.0 if str(r["ground_truth"]).lower() == "true" else 0.0 for r in chunk_valid]
            steps.append(s)
            briers.append(np.mean([r["brier_score"] for r in chunk_valid]))
            eces.append(compute_ece(p_yes, actual))
            accs.append(np.mean([1.0 if (p > 0.5) == (a == 1.0) else 0.0 for p, a in zip(p_yes, actual)]))
        return steps, briers, eces, accs

    ind_steps, ind_briers, ind_eces, ind_accs = per_step_metrics(records, step_size)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0][0]
    ax.plot(ind_steps, ind_briers, "o-", color="#2563eb", linewidth=2, markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean Brier score")
    ax.set_title("Brier Score Over Training")
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    ax.plot(ind_steps, [1 - b for b in ind_briers], "o-", color="#2563eb", linewidth=2, markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward (1 - Brier)")
    ax.set_title("Reward Over Training")
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    ax.plot(ind_steps, ind_eces, "o-", color="#2563eb", linewidth=2, markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("ECE")
    ax.set_title("Expected Calibration Error Over Training")
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    ax.plot(ind_steps, ind_accs, "o-", color="#2563eb", linewidth=2, markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Over Training (p>0.5 → YES)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = results_dir / "brier_over_time.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # --- Per-generation CSV ---
    csv_path = results_dir / "calibration_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "step", "reward", "brier_score", "p_yes", "ground_truth", "parseable"])
        for i, r in enumerate(records):
            writer.writerow([
                i, i // step_size,
                r.get("reward", ""), r.get("brier_score", ""),
                r.get("p_yes", ""), r.get("ground_truth", ""),
                1 if r.get("p_yes") is not None else 0,
            ])
    print(f"Saved {csv_path}")



if __name__ == "__main__":
    main()
