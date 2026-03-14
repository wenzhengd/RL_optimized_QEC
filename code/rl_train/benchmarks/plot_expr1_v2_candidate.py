"""Plot Expr1 V2 candidate sweep with the same high-level style as V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _aggregate_fixed_zero_success(stage_dir: Path) -> float:
    vals: list[float] = []
    for path in sorted(stage_dir.glob("seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fixed_zero = payload.get("eval_metrics", {}).get("fixed_zero", {})
        success_rate = fixed_zero.get("success_rate")
        if success_rate is not None:
            vals.append(float(success_rate))
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=float)))


def _load_expr1_v2_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for stage_name, stage_data in payload.get("stages", {}).items():
        cfg = stage_data.get("config_overrides", {})
        agg = stage_data.get("aggregate", {})
        regime_a = float(cfg.get("steane_channel_regime_a", 0.0))
        regime_b = float(cfg.get("steane_channel_regime_b", 0.0))
        ratio_label = "1q/2q = 1/1" if abs(regime_a - regime_b) < 1e-12 else "1q/2q = 1/10"
        rows.append(
            {
                "stage_name": stage_name,
                "scale": regime_a,
                "ratio_label": ratio_label,
                "learned_success_mean": float(agg.get("learned_policy", {}).get("success_rate_mean", float("nan"))),
                "improve_ler_mean": float(
                    agg.get("improvement_vs_fixed_zero", {}).get("ler_proxy_mean", float("nan"))
                ),
                "fixed_zero_success_mean": _aggregate_fixed_zero_success(summary_path.parent / stage_name),
            }
        )
    return rows


def _plot_expr1_v2(rows: list[dict[str, Any]], out_path: Path, figure_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    specs = [
        ("1q/2q = 1/1", lambda r: r["ratio_label"] == "1q/2q = 1/1"),
        ("1q/2q = 1/10", lambda r: r["ratio_label"] == "1q/2q = 1/10"),
    ]
    abs_colors = {"learned": "#2a9d8f", "fixed_zero": "#b08968"}
    rel_color = "#d62828"
    for ax, (title, pred) in zip(axes, specs):
        subset = sorted([r for r in rows if pred(r)], key=lambda r: float(r["scale"]))
        xpos = np.arange(len(subset), dtype=float)
        width = 0.34
        learned = [100.0 * float(r["learned_success_mean"]) for r in subset]
        fixed_zero = [100.0 * float(r["fixed_zero_success_mean"]) for r in subset]
        improve = [100.0 * float(r["improve_ler_mean"]) for r in subset]
        tick_labels = [f"{float(r['scale']):g}" for r in subset]

        ax.bar(xpos - width / 2.0, learned, width=width, color=abs_colors["learned"], label="learned success")
        ax.bar(xpos + width / 2.0, fixed_zero, width=width, color=abs_colors["fixed_zero"], label="fixed_zero success")
        ax.set_xticks(xpos)
        ax.set_xticklabels(tick_labels)
        ax.set_title(title)
        ax.set_xlabel("overall gate-noise scale")
        ax.set_ylabel("absolute success rate (%)", color="#264653")
        ax.set_ylim(50.0, 100.5)
        ax.tick_params(axis="y", labelcolor="#264653")
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        ax_rel = ax.twinx()
        ax_rel.plot(
            xpos,
            improve,
            marker="o",
            markersize=6.0,
            linewidth=2.2,
            color=rel_color,
            label="improvement vs fixed_zero (LER~)",
        )
        ax_rel.axhline(0.0, color=rel_color, linewidth=0.9, alpha=0.45, linestyle=":")
        ax_rel.set_ylabel("relative improve(LER~) (%)", color=rel_color)
        ax_rel.tick_params(axis="y", labelcolor=rel_color)

        if ax is axes[1]:
            handles_abs, labels_abs = ax.get_legend_handles_labels()
            handles_rel, labels_rel = ax_rel.get_legend_handles_labels()
            ax.legend(handles_abs + handles_rel, labels_abs + labels_rel, loc="best")

    fig.suptitle(figure_title, fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Expr1 V2 candidate sweep.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr1_gate_only_v2/phaseA_candidate/summary.json",
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr1_gate_only_v2/phaseA_candidate/expr1_v2_phaseA_candidate.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--figure-title",
        type=str,
        default="Expr1 V2 Phase A Candidate Sweep",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    out_path = Path(args.output_path)
    rows = _load_expr1_v2_rows(summary_path)
    _plot_expr1_v2(rows, out_path, figure_title=str(args.figure_title))
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
