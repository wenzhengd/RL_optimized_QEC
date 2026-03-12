"""Plot core figures for the RL Steane tuning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_stage_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for stage_name, stage_data in payload.get("stages", {}).items():
        cfg = stage_data.get("config_overrides", {})
        agg = stage_data.get("aggregate", {})
        rows.append(
            {
                "stage_name": stage_name,
                "regime_a": float(cfg.get("steane_channel_regime_a", 0.0)),
                "regime_b": float(cfg.get("steane_channel_regime_b", 0.0)),
                "f_hz": float(cfg.get("steane_channel_corr_f", 0.0)),
                "g": float(cfg.get("steane_channel_corr_g", 0.0)),
                "p_meas": float(cfg.get("steane_measurement_bitflip_prob", 0.0)),
                "improve_ler_mean": float(agg.get("improvement_vs_fixed_zero", {}).get("ler_proxy_mean", float("nan"))),
                "learned_success_mean": float(agg.get("learned_policy", {}).get("success_rate_mean", float("nan"))),
            }
        )
    return rows


def _aggregate_cycle_files(files: list[Path]) -> dict[tuple[str, int], tuple[float, float]]:
    values: dict[tuple[str, int], list[float]] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for policy_key in ("primary_policy", "secondary_policy"):
            policy = payload[policy_key]
            label = str(policy["label"])
            for row in policy["cycle_sweep"]:
                key = (label, int(row["n_rounds"]))
                values.setdefault(key, []).append(float(row["learned"]["logical_observable_proxy"]))
    out: dict[tuple[str, int], tuple[float, float]] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = (float(np.mean(arr)), float(np.std(arr)))
    return out


def _plot_expr1_phasea(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    specs = [
        ("1q/2q = 1/4", lambda r: abs((r["regime_b"] / max(r["regime_a"], 1e-12)) - 4.0) < 1e-9, "regime_a"),
        ("1q/2q = 4/1", lambda r: abs((r["regime_a"] / max(r["regime_b"], 1e-12)) - 4.0) < 1e-9, "regime_b"),
    ]
    for ax, (title, pred, xkey) in zip(axes, specs):
        subset = sorted([r for r in rows if pred(r)], key=lambda r: float(r[xkey]))
        xs = [float(r[xkey]) for r in subset]
        succ = [100.0 * float(r["learned_success_mean"]) for r in subset]
        imp = [100.0 * float(r["improve_ler_mean"]) for r in subset]
        ax.plot(xs, succ, marker="o", linewidth=2.0, label="learned success (%)")
        ax.plot(xs, imp, marker="s", linewidth=2.0, label="improve(LER~) (%)")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("overall gate-noise scale")
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("percent")
    axes[1].legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _heatmap(
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    out_path: Path,
    p_meas_values: list[float] | None = None,
) -> None:
    if p_meas_values is None:
        p_meas_values = [0.0]
    f_vals = sorted({float(r["f_hz"]) for r in rows})
    g_vals = sorted({float(r["g"]) for r in rows})
    ncols = len(p_meas_values)
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 4.6), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for ax, p_meas in zip(axes, p_meas_values):
        mat = np.full((len(f_vals), len(g_vals)), np.nan, dtype=float)
        for r in rows:
            if abs(float(r["p_meas"]) - float(p_meas)) > 1e-12:
                continue
            i = f_vals.index(float(r["f_hz"]))
            j = g_vals.index(float(r["g"]))
            mat[i, j] = 100.0 * float(r[value_key])
        vmax = max(5.0, float(np.nanmax(np.abs(mat))))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(g_vals)))
        ax.set_xticklabels([f"{g:.1f}" for g in g_vals])
        ax.set_yticks(range(len(f_vals)))
        ax.set_yticklabels([f"{f:.0e}" for f in f_vals])
        ax.set_xlabel("g")
        ax.set_ylabel("f (Hz)")
        ax.set_title(title if ncols == 1 else f"{title}\np_meas={p_meas:.2f}")
        for i in range(len(f_vals)):
            for j in range(len(g_vals)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:+.1f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_expr4_cycle(cycle_stats: dict[tuple[str, int], tuple[float, float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    rounds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for label, color in (
        ("trained_on_full_composite", "#1d3557"),
        ("trained_on_composite", "#e76f51"),
    ):
        means = [cycle_stats[(label, r)][0] for r in rounds]
        stds = [cycle_stats[(label, r)][1] for r in rounds]
        ax.plot(rounds, means, marker="o", linewidth=2.2, label=label, color=color)
        ax.fill_between(rounds, np.asarray(means) - np.asarray(stds), np.asarray(means) + np.asarray(stds), alpha=0.18, color=color)
    ax.set_title("Expr4 Phase C: Cycle-Decay Showcase")
    ax.set_xlabel("n_rounds")
    ax.set_ylabel("logical_observable_proxy")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RL Steane tuning experiment figures.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="code/data_generated/rl_Steane_tune_experiment",
        help="Base experiment directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Default: <base-dir>/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    out_dir = Path(args.output_dir) if str(args.output_dir).strip() else (base / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    expr1_rows = _load_stage_rows(base / "expr1_gate_only/phaseA_quick/summary.json")
    expr2_rows = _load_stage_rows(base / "expr2_standard_composite/phaseA_quick/summary.json")
    expr3_rows = _load_stage_rows(base / "expr3_full_composite/phaseA_quick/summary.json")
    expr4_cycle = _aggregate_cycle_files(
        sorted((base / "expr4_cycle_decay_full_composite").glob("phaseC_showcase_p001_f1e2_g16_seed*.json"))
    )

    _plot_expr1_phasea(expr1_rows, out_dir / "expr1_phaseA_lines.png")
    _heatmap(expr2_rows, "improve_ler_mean", "Expr2 Phase A: improve(LER~) (%)", out_dir / "expr2_phaseA_heatmap.png")
    _heatmap(
        expr3_rows,
        "improve_ler_mean",
        "Expr3 Phase A: improve(LER~) (%)",
        out_dir / "expr3_phaseA_heatmaps.png",
        p_meas_values=[0.01, 0.02, 0.05],
    )
    _plot_expr4_cycle(expr4_cycle, out_dir / "expr4_phaseC_cycle_decay.png")

    for name in (
        "expr1_phaseA_lines.png",
        "expr2_phaseA_heatmap.png",
        "expr3_phaseA_heatmaps.png",
        "expr4_phaseC_cycle_decay.png",
    ):
        print(f"Saved figure: {out_dir / name}")


if __name__ == "__main__":
    main()
