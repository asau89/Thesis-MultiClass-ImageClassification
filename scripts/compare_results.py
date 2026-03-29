"""
compare_results.py — Post-tuning analysis across all phases.

Reads all tuning_results/phase_*/phase_summary.json files, prints a combined
comparison table, identifies the overall best configuration, and generates
summary charts.

Usage:
  python compare_results.py
"""
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.config import Config


def load_all_results(tuning_dir: Path) -> dict[int, list[dict]]:
    """Returns {phase_number: [trial_result, ...]} for all completed phases."""
    all_results: dict[int, list] = {}
    for phase_dir in sorted(tuning_dir.glob("phase_*")):
        summary_path = phase_dir / "phase_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        phase_num = summary.get("phase", int(phase_dir.name.split("_")[1]))
        all_results[phase_num] = summary.get("ranked_results", [])
    return all_results


def print_combined_table(all_results: dict[int, list]) -> tuple[dict, float]:
    """Prints a combined ranking table across all phases. Returns (best_config, best_acc)."""
    all_valid = []
    for phase, results in sorted(all_results.items()):
        for r in results:
            if r.get("best_val_acc", -1) >= 0:
                all_valid.append({"phase": phase, **r})

    all_valid.sort(key=lambda r: r["best_val_acc"], reverse=True)

    print("\n" + "═" * 90)
    print("HYPERPARAMETER TUNING — COMBINED RESULTS (all phases, ranked by val accuracy)")
    print("═" * 90)
    print(f"{'Rank':<5} {'Phase':<7} {'Trial':<12} {'Val Acc':>9} {'F1':>8} {'Prec':>8} {'Recall':>8}  Config")
    print("─" * 90)
    for rank, r in enumerate(all_valid, 1):
        hp_str = "  ".join(f"{k}={v}" for k, v in r["hyperparams"].items())
        print(
            f"{rank:<5} {r['phase']:<7} {r['trial_name']:<12} "
            f"{r['best_val_acc']:>8.2f}% "
            f"{r['best_val_f1']:>7.2f}% "
            f"{r['best_val_precision']:>7.2f}% "
            f"{r['best_val_recall']:>7.2f}%  "
            f"{hp_str}"
        )

    if all_valid:
        best = all_valid[0]
        print(f"\n★ BEST OVERALL: Phase {best['phase']} | {best['trial_name']} "
              f"| Val Acc = {best['best_val_acc']:.2f}%  F1 = {best['best_val_f1']:.2f}%")
        print(f"  Config: {best['hyperparams']}")
        return best["hyperparams"], best["best_val_acc"]
    return {}, 0.0


def save_best_config(best_config: dict, best_acc: float, tuning_dir: Path):
    """Saves the best config as JSON for use in final training."""
    out = {
        "best_val_acc": best_acc,
        "best_config": best_config,
        "how_to_use": (
            "Copy the values from best_config into config.py before running train.py "
            "for your final full-accuracy training run."
        ),
    }
    out_path = tuning_dir / "best_config.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[compare] Best config saved → {out_path}")


def save_per_phase_bar_chart(all_results: dict[int, list], tuning_dir: Path):
    """
    One bar chart per phase, arranged in a grid. Shows val acc, F1, precision, recall.
    All charts saved as a single combined PNG.
    """
    phases = sorted(all_results.keys())
    if not phases:
        return

    n = len(phases)
    fig = plt.figure(figsize=(7 * n, 6))
    gs  = gridspec.GridSpec(1, n, figure=fig)

    metric_keys   = ["best_val_acc", "best_val_f1", "best_val_precision", "best_val_recall"]
    metric_labels = ["Val Acc (%)", "F1 Macro (%)", "Precision (%)", "Recall (%)"]
    colors        = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for col, phase in enumerate(phases):
        ax = fig.add_subplot(gs[0, col])
        results = all_results[phase]
        trial_names = [r["trial_name"] for r in results]
        x = np.arange(len(trial_names))
        width = 0.2

        for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
            values = [r.get(key, 0) for r in results]
            ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(trial_names, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"Phase {phase}", fontsize=13)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Hyperparameter Tuning — All Phases", fontsize=15, y=1.02)
    plt.tight_layout()
    out_path = tuning_dir / "all_phases_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[compare] All-phases chart saved → {out_path}")


def save_accuracy_trend_chart(all_results: dict[int, list], tuning_dir: Path):
    """
    Line chart showing the best val accuracy achieved in each trial, in order,
    across all phases. Useful for spotting the overall improvement trajectory.
    """
    trial_labels = []
    accuracies   = []
    f1_scores    = []

    for phase in sorted(all_results.keys()):
        for r in all_results[phase]:
            if r.get("best_val_acc", -1) >= 0:
                trial_labels.append(f"P{phase}-{r['trial_name']}")
                accuracies.append(r["best_val_acc"])
                f1_scores.append(r["best_val_f1"])

    if not trial_labels:
        return

    x = np.arange(len(trial_labels))
    fig, ax = plt.subplots(figsize=(max(10, len(trial_labels) * 1.2), 5))
    ax.plot(x, accuracies, marker="o", label="Val Accuracy (%)", color="#4C72B0", linewidth=2)
    ax.plot(x, f1_scores,  marker="s", label="Val F1 Macro (%)",  color="#DD8452", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy + F1 Trend Across All Trials")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = tuning_dir / "accuracy_trend.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"[compare] Accuracy trend chart saved → {out_path}")


def main():
    cfg = Config()
    tuning_dir = Path(cfg.tuning_dir)

    if not tuning_dir.exists():
        print(f"[compare] Tuning results directory not found: {tuning_dir}")
        print("          Run `python tune.py --phase 1` first.")
        return

    all_results = load_all_results(tuning_dir)
    if not all_results:
        print("[compare] No completed phase summaries found.")
        return

    print(f"[compare] Found results for phase(s): {sorted(all_results.keys())}")

    best_config, best_acc = print_combined_table(all_results)
    save_best_config(best_config, best_acc, tuning_dir)
    save_per_phase_bar_chart(all_results, tuning_dir)
    save_accuracy_trend_chart(all_results, tuning_dir)

    print("\n[compare] All outputs written to:", tuning_dir)


if __name__ == "__main__":
    main()
