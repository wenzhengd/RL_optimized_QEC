"""Plot delta-style summaries for Expr1->Expr2 transfer evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot transfer delta comparison figures.")
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def _label(row: dict[str, float]) -> str:
    return f"s={float(row['scale']):g}, f={float(row['f_hz']):.0f}, g={float(row['g']):.1f}"


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    targets = payload["targets"]

    rows = [
        row
        for _, row in sorted(
            targets.items(),
            key=lambda kv: float(kv[1]["target_learned_success_rate"]),
            reverse=True,
        )
    ]

    labels = [_label(row) for row in rows]
    improve_mean = [100.0 * float(row["transfer_improve_vs_fixed_zero_mean"]) for row in rows]
    improve_std = [100.0 * float(row["transfer_improve_vs_fixed_zero_std"]) for row in rows]
    delta_success_mean = [100.0 * float(row["delta_success_vs_target_mean"]) for row in rows]
    delta_success_std = [100.0 * float(row["delta_success_vs_target_std"]) for row in rows]

    x = np.arange(len(labels), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.2, 5.4), constrained_layout=True)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.bar(
        x - width / 2.0,
        improve_mean,
        width=width,
        color="#457b9d",
        label="transfer vs fixed_zero: improve(LER~) (%)",
    )
    ax.bar(
        x + width / 2.0,
        delta_success_mean,
        width=width,
        color="#c8553d",
        label="transfer vs Expr2-trained: delta success (%)",
    )
    ax.errorbar(x - width / 2.0, improve_mean, yerr=improve_std, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    ax.errorbar(
        x + width / 2.0,
        delta_success_mean,
        yerr=delta_success_std,
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.0,
    )

    all_bounds = np.concatenate(
        [
            np.asarray(improve_mean, dtype=float) - np.asarray(improve_std, dtype=float),
            np.asarray(improve_mean, dtype=float) + np.asarray(improve_std, dtype=float),
            np.asarray(delta_success_mean, dtype=float) - np.asarray(delta_success_std, dtype=float),
            np.asarray(delta_success_mean, dtype=float) + np.asarray(delta_success_std, dtype=float),
            np.asarray([0.0], dtype=float),
        ]
    )
    lower = float(np.min(all_bounds))
    upper = float(np.max(all_bounds))
    span = max(upper - lower, 4.0)
    pad = max(0.15 * span, 1.0)

    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("delta (%)")
    ax.set_title("Expr1->Expr2 Transfer Deltas")
    ax.set_ylim(lower - pad, upper + pad)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="best")

    for xpos, val in zip(x - width / 2.0, improve_mean):
        ax.text(xpos, val + (0.5 if val >= 0 else -0.8), f"{val:+.2f}", ha="center", va="bottom", fontsize=8)
    for xpos, val in zip(x + width / 2.0, delta_success_mean):
        ax.text(xpos, val + (0.5 if val >= 0 else -0.8), f"{val:+.2f}", ha="center", va="bottom", fontsize=8)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
