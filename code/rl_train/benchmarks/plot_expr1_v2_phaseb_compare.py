"""Plot old vs stronger-training Expr1 V2 Phase B results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _ratio_label(stage_name: str) -> str:
    if "ratio_1to1" in stage_name:
        return "1q/2q = 1/1"
    if "ratio_1to10" in stage_name:
        return "1q/2q = 1/10"
    raise ValueError(f"Unrecognized stage name: {stage_name}")


def _scale_value(stage_name: str) -> float:
    tail = stage_name.split("_scale_")[-1]
    return float(tail.replace("_", "."))


def _load_rows(summary_path: Path, label: str) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for stage_name, stage_data in payload.get("stages", {}).items():
        agg = stage_data.get("aggregate", {})
        rows.append(
            {
                "run_label": label,
                "stage_name": stage_name,
                "ratio_label": _ratio_label(stage_name),
                "scale": _scale_value(stage_name),
                "learned_success_mean": float(agg.get("learned_policy", {}).get("success_rate_mean", float("nan"))),
                "improve_ler_mean": float(
                    agg.get("improvement_vs_fixed_zero", {}).get("ler_proxy_mean", float("nan"))
                ),
            }
        )
    return rows


def _plot(rows: list[dict[str, Any]], out_path: Path, figure_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    run_colors = {"old": "#6c757d", "stronger": "#2a9d8f"}
    run_offsets = {"old": -0.18, "stronger": 0.18}
    width = 0.32

    for ax, ratio_label in zip(axes, ("1q/2q = 1/1", "1q/2q = 1/10")):
        subset = sorted([r for r in rows if r["ratio_label"] == ratio_label], key=lambda r: float(r["scale"]))
        scales = sorted({float(r["scale"]) for r in subset})
        x = np.arange(len(scales), dtype=float)

        for run_label in ("old", "stronger"):
            by_scale = {float(r["scale"]): r for r in subset if r["run_label"] == run_label}
            vals = [100.0 * by_scale[s]["learned_success_mean"] for s in scales if s in by_scale]
            xpos = np.asarray([x[i] + run_offsets[run_label] for i, s in enumerate(scales) if s in by_scale], dtype=float)
            if len(vals) != len(xpos) or not vals:
                continue
            ax.bar(xpos, vals, width=width, color=run_colors[run_label], alpha=0.9, label=f"{run_label} success")

        ax.set_title(ratio_label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:g}" for s in scales])
        ax.set_xlabel("overall gate-noise scale")
        ax.set_ylabel("learned success rate (%)", color="#264653")
        ax.tick_params(axis="y", labelcolor="#264653")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_ylim(50.0, 100.5)

        ax_rel = ax.twinx()
        for run_label, marker in (("old", "o"), ("stronger", "s")):
            by_scale = {float(r["scale"]): r for r in subset if r["run_label"] == run_label}
            xpos = np.asarray([x[i] for i, s in enumerate(scales) if s in by_scale], dtype=float)
            vals = [100.0 * by_scale[s]["improve_ler_mean"] for s in scales if s in by_scale]
            if len(vals) != len(xpos) or not vals:
                continue
            ax_rel.plot(
                xpos,
                vals,
                marker=marker,
                markersize=6.0,
                linewidth=2.0,
                color=run_colors[run_label],
                label=f"{run_label} improve(LER~)",
            )
        ax_rel.axhline(0.0, color="#b22222", linewidth=0.9, alpha=0.4, linestyle=":")
        ax_rel.set_ylabel("improve(LER~) (%)")

        if ax is axes[1]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_rel.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best")

    fig.suptitle(figure_title, fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Expr1 V2 Phase B old vs stronger comparison.")
    parser.add_argument(
        "--old-summary-json",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr1_gate_only_v2/phaseB_focused/summary.json",
    )
    parser.add_argument(
        "--new-summary-json",
        type=str,
        default=(
            "code/data_generated/rl_steane_tune_experiments_V2/"
            "expr1_gate_only_v2/phaseB_focused_stronger_train/summary.json"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=(
            "code/data_generated/rl_steane_tune_experiments_V2/"
            "expr1_gate_only_v2/phaseB_focused_stronger_train/expr1_v2_phaseB_old_vs_stronger.png"
        ),
    )
    parser.add_argument(
        "--figure-title",
        type=str,
        default="Expr1 V2 Phase B: Old vs Stronger Training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.old_summary_json), label="old")
    rows.extend(_load_rows(Path(args.new_summary_json), label="stronger"))
    _plot(rows, Path(args.output_path), figure_title=str(args.figure_title))
    print(f"Saved figure: {args.output_path}")


if __name__ == "__main__":
    main()
