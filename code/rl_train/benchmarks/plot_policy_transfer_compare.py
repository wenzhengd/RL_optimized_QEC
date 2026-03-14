"""Plot transfer-vs-fixed-zero-vs-Expr2-trained comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot transfer comparison figures.")
    parser.add_argument(
        "--summary-json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    targets = payload["targets"]
    labels = []
    fixed = []
    target = []
    transfer_mean = []
    transfer_std = []

    for target_label, row in sorted(
        targets.items(),
        key=lambda kv: float(kv[1]["target_learned_success_rate"]),
        reverse=True,
    ):
        labels.append(
            f"s={float(row['scale']):g}, f={float(row['f_hz']):.0f}, g={float(row['g']):.1f}"
        )
        fixed.append(100.0 * float(row["fixed_zero_success_rate"]))
        target.append(100.0 * float(row["target_learned_success_rate"]))
        # Recover transfer mean success from report-level rows for this target is not in target summary,
        # so approximate from target success + mean delta.
        transfer_mean.append(100.0 * (float(row["target_learned_success_rate"]) + float(row["delta_success_vs_target_mean"])))
        transfer_std.append(100.0 * float(row["delta_success_vs_target_std"]))

    x = np.arange(len(labels), dtype=float)
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.bar(x - width, fixed, width=width, color="#b08968", label="fixed_zero")
    ax.bar(x, transfer_mean, width=width, color="#457b9d", label="Expr1-trained transfer")
    ax.bar(x + width, target, width=width, color="#2a9d8f", label="Expr2-trained")
    ax.errorbar(x, transfer_mean, yerr=transfer_std, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    fixed_arr = np.asarray(fixed, dtype=float)
    transfer_arr = np.asarray(transfer_mean, dtype=float)
    transfer_std_arr = np.asarray(transfer_std, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    lower_bound = float(
        np.min(
            np.concatenate(
                [
                    fixed_arr,
                    transfer_arr - transfer_std_arr,
                    target_arr,
                ]
            )
        )
    )
    upper_bound = float(
        np.max(
            np.concatenate(
                [
                    fixed_arr,
                    transfer_arr + transfer_std_arr,
                    target_arr,
                ]
            )
        )
    )
    span = max(upper_bound - lower_bound, 0.6)
    pad = max(0.12 * span, 0.08)
    ymin = max(0.0, lower_bound - pad)
    ymax = min(100.0, upper_bound + pad)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Expr1 Transfer vs Fixed-Zero vs Expr2-Trained")
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    for xpos, val in zip(x - width, fixed):
        ax.text(xpos, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for xpos, val in zip(x, transfer_mean):
        ax.text(xpos, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for xpos, val in zip(x + width, target):
        ax.text(xpos, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
