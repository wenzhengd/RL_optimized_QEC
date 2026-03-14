"""Plot clear Expr4 V2 cycle-decay curves from aggregated summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _rows_by_label(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("rows", []):
        grouped.setdefault(str(row["policy_label"]), []).append(row)
    for label in grouped:
        grouped[label].sort(key=lambda row: int(row["n_rounds"]))
    return grouped


def _plot_metric(ax, rows: list[dict[str, Any]], label: str, color: str, mean_key: str, std_key: str, ylabel: str) -> None:
    rounds = np.asarray([int(row["n_rounds"]) for row in rows], dtype=int)
    means = np.asarray([float(row.get(mean_key, np.nan)) for row in rows], dtype=float)
    stds = np.asarray([float(row.get(std_key, 0.0)) for row in rows], dtype=float)
    ax.plot(rounds, means, marker="o", linewidth=2.2, color=color, label=label)
    ax.fill_between(rounds, means - stds, means + stds, color=color, alpha=0.16)
    ax.set_xlabel("QEC cycles (n_rounds)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Expr4 V2 cycle-decay summary.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/summary.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/expr4_v2_cycle_decay.png",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    grouped = _rows_by_label(payload)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    colors = {
        payload.get("primary_policy_label"): "#1d3557",
        payload.get("transfer_policy_label"): "#457b9d",
        "fixed_zero": "#8d6e63",
    }
    labels = [payload.get("primary_policy_label"), payload.get("transfer_policy_label"), "fixed_zero"]

    for label in labels:
        if not label or label not in grouped:
            continue
        _plot_metric(
            axes[0],
            grouped[label],
            label,
            colors.get(label, "#333333"),
            "success_rate_mean",
            "success_rate_std",
            "Success rate",
        )
        _plot_metric(
            axes[1],
            grouped[label],
            label,
            colors.get(label, "#333333"),
            "logical_observable_proxy_mean",
            "logical_observable_proxy_std",
            "Logical observable proxy",
        )

    axes[0].set_title("Expr4 V2: Success Rate vs Cycle Count")
    axes[1].set_title("Expr4 V2: Logical Proxy vs Cycle Count")
    axes[0].legend(loc="best")
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
