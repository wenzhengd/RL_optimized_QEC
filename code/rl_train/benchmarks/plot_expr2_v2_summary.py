"""Plot Expr2 V2 phase summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for stage_name, stage_data in payload.get("stages", {}).items():
        cfg = stage_data.get("config_overrides", {})
        agg = stage_data.get("aggregate", {})
        runs = stage_data.get("runs", [])
        pos = sum(1 for r in runs if float(r["improvement_vs_fixed_zero"]["ler_proxy"]) > 0.0)
        rows.append(
            {
                "stage_name": stage_name,
                "scale": float(cfg.get("steane_channel_regime_a", 0.0)),
                "f_hz": float(cfg.get("steane_channel_corr_f", 0.0)),
                "g": float(cfg.get("steane_channel_corr_g", 0.0)),
                "improve_mean": float(agg.get("improvement_vs_fixed_zero", {}).get("ler_proxy_mean", float("nan"))),
                "improve_std": float(agg.get("improvement_vs_fixed_zero", {}).get("ler_proxy_std", float("nan"))),
                "learned_success_mean": float(agg.get("learned_policy", {}).get("success_rate_mean", float("nan"))),
                "pos_ratio": float(pos / len(runs)) if runs else float("nan"),
            }
        )
    return rows


def _fmt_freq(f_hz: float) -> str:
    if f_hz <= 0:
        return "0"
    exp = int(round(np.log10(f_hz)))
    if abs((10**exp) - f_hz) / max(1.0, f_hz) < 1e-9:
        return f"1e{exp}"
    return f"{f_hz:.3g}"


def _plot_phasea(rows: list[dict[str, Any]], out_path: Path) -> None:
    scale_vals = sorted({float(r["scale"]) for r in rows})
    f_vals = sorted({float(r["f_hz"]) for r in rows})
    g_vals = sorted({float(r["g"]) for r in rows})

    fig, axes = plt.subplots(1, len(scale_vals), figsize=(6.2 * len(scale_vals), 5.0), constrained_layout=True)
    if len(scale_vals) == 1:
        axes = [axes]

    for ax, scale in zip(axes, scale_vals):
        mat = np.full((len(f_vals), len(g_vals)), np.nan, dtype=float)
        for row in rows:
            if abs(float(row["scale"]) - scale) > 1e-12:
                continue
            i = f_vals.index(float(row["f_hz"]))
            j = g_vals.index(float(row["g"]))
            mat[i, j] = 100.0 * float(row["improve_mean"])
        vmax = max(5.0, float(np.nanmax(np.abs(mat))))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Expr2 V2 Phase A\nscale={scale:g}")
        ax.set_xticks(range(len(g_vals)), [f"{g:.1f}" for g in g_vals])
        ax.set_yticks(range(len(f_vals)), [_fmt_freq(f) for f in f_vals])
        ax.set_xlabel("g")
        ax.set_ylabel("f (Hz)")
        for i in range(len(f_vals)):
            for j in range(len(g_vals)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_phase_bc(rows_b: list[dict[str, Any]], rows_c: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    for ax, rows, title in (
        (axes[0], rows_b, "Expr2 V2 Phase B Focused"),
        (axes[1], rows_c, "Expr2 V2 Phase C Confirm"),
    ):
        rows_sorted = sorted(rows, key=lambda r: float(r["improve_mean"]), reverse=True)
        labels = [f"s={r['scale']:g}, f={_fmt_freq(r['f_hz'])}, g={r['g']:.1f}" for r in rows_sorted]
        x = np.arange(len(rows_sorted), dtype=float)
        imp = 100.0 * np.asarray([float(r["improve_mean"]) for r in rows_sorted], dtype=float)
        err = 100.0 * np.asarray([float(r["improve_std"]) for r in rows_sorted], dtype=float)
        pos = 100.0 * np.asarray([float(r["pos_ratio"]) for r in rows_sorted], dtype=float)
        succ = 100.0 * np.asarray([float(r["learned_success_mean"]) for r in rows_sorted], dtype=float)

        ax.bar(x, imp, color="#2a9d8f", alpha=0.88)
        ax.errorbar(x, imp, yerr=err, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel("improve(LER~) (%)")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        for i in range(len(rows_sorted)):
            ax.text(i, imp[i] + max(1.5, 0.08 * np.max(imp)), f"{pos[i]:.0f}% pos\n{succ[i]:.1f}% succ", ha="center", fontsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Expr2 V2 summaries.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr2_standard_composite_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    out_dir = Path(args.output_dir) if str(args.output_dir).strip() else (base / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    phasea_rows = _load_rows(base / "phaseA_pilot_balanced" / "summary.json")
    phaseb_rows = _load_rows(base / "phaseB_focused" / "summary.json")
    phasec_rows = _load_rows(base / "phaseC_confirm" / "summary.json")

    phasea_out = out_dir / "expr2_v2_phaseA_heatmaps.png"
    phasebc_out = out_dir / "expr2_v2_phaseBC_compare.png"
    _plot_phasea(phasea_rows, phasea_out)
    _plot_phase_bc(phaseb_rows, phasec_rows, phasebc_out)
    print(f"Saved figure: {phasea_out}")
    print(f"Saved figure: {phasebc_out}")


if __name__ == "__main__":
    main()
