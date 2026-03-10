"""Plot RL advantage figures for composed correlated (f, g) benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _mean_std(values: List[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _load_stage_rows(summary_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    stages = payload.get("stages", {})
    rows: List[Dict[str, Any]] = []
    for stage_name, stage_data in stages.items():
        cfg = stage_data.get("config_overrides", {})
        runs = stage_data.get("runs", [])
        if not runs:
            continue
        f_hz = float(cfg.get("steane_channel_corr_f", 0.0))
        g = float(cfg.get("steane_channel_corr_g", 0.0))
        improve_ler = [float(r["improvement_vs_fixed_zero"]["ler_proxy"]) for r in runs]
        learned_success = [float(r["eval_metrics"]["learned"]["success_rate"]) for r in runs]
        fixed_success = [float(r["eval_metrics"]["fixed_zero"]["success_rate"]) for r in runs]
        delta_success = [l - f for l, f in zip(learned_success, fixed_success)]
        improve_mean, improve_std = _mean_std(improve_ler)
        delta_mean, delta_std = _mean_std(delta_success)
        pos_ratio = float(np.mean(np.asarray(delta_success, dtype=float) > 0.0))
        rows.append(
            {
                "stage_name": str(stage_name),
                "f_hz": f_hz,
                "g": g,
                "n": int(len(runs)),
                "improve_mean": improve_mean,
                "improve_std": improve_std,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "pos_ratio": pos_ratio,
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


def _plot_phase1_heatmaps(rows: List[Dict[str, Any]], out_path: Path) -> None:
    f_vals = sorted({float(r["f_hz"]) for r in rows})
    g_vals = sorted({float(r["g"]) for r in rows})
    f_idx = {v: i for i, v in enumerate(f_vals)}
    g_idx = {v: i for i, v in enumerate(g_vals)}

    improve = np.full((len(f_vals), len(g_vals)), np.nan, dtype=float)
    delta = np.full((len(f_vals), len(g_vals)), np.nan, dtype=float)
    posr = np.full((len(f_vals), len(g_vals)), np.nan, dtype=float)
    for r in rows:
        i = f_idx[float(r["f_hz"])]
        j = g_idx[float(r["g"])]
        improve[i, j] = 100.0 * float(r["improve_mean"])
        delta[i, j] = 100.0 * float(r["delta_mean"])
        posr[i, j] = 100.0 * float(r["pos_ratio"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)

    vmax1 = max(5.0, float(np.nanmax(np.abs(improve))))
    im1 = axes[0].imshow(improve, cmap="RdYlGn", vmin=-vmax1, vmax=vmax1, aspect="auto")
    axes[0].set_title("Phase1: improve(LER~) vs fixed-zero (%)")
    axes[0].set_xticks(range(len(g_vals)), [f"{g:.1f}" for g in g_vals])
    axes[0].set_yticks(range(len(f_vals)), [_fmt_freq(f) for f in f_vals])
    axes[0].set_xlabel("correlated strength g")
    axes[0].set_ylabel("correlated frequency f (Hz)")
    for i in range(len(f_vals)):
        for j in range(len(g_vals)):
            if np.isfinite(improve[i, j]):
                axes[0].text(
                    j,
                    i,
                    f"{improve[i, j]:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    vmax2 = max(1.0, float(np.nanmax(np.abs(delta))))
    im2 = axes[1].imshow(delta, cmap="PuOr", vmin=-vmax2, vmax=vmax2, aspect="auto")
    axes[1].set_title("Phase1: delta success (learned - fixed, pp)")
    axes[1].set_xticks(range(len(g_vals)), [f"{g:.1f}" for g in g_vals])
    axes[1].set_yticks(range(len(f_vals)), [_fmt_freq(f) for f in f_vals])
    axes[1].set_xlabel("correlated strength g")
    axes[1].set_ylabel("correlated frequency f (Hz)")
    for i in range(len(f_vals)):
        for j in range(len(g_vals)):
            if np.isfinite(delta[i, j]):
                axes[1].text(
                    j,
                    i,
                    f"{delta[i, j]:+.2f}\n({posr[i, j]:.0f}%)",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="black",
                )
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_phase2_scatter(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    xs = np.asarray([np.log10(float(r["f_hz"])) for r in rows], dtype=float)
    ys = np.asarray([float(r["g"]) for r in rows], dtype=float)
    cs = np.asarray([100.0 * float(r["improve_mean"]) for r in rows], dtype=float)
    ss = 160.0 + 520.0 * np.asarray([float(r["pos_ratio"]) for r in rows], dtype=float)
    sc = ax.scatter(xs, ys, c=cs, s=ss, cmap="RdYlGn", edgecolors="black", linewidths=0.8)
    ax.set_title("Phase2 Pilot: RL advantage over (f, g)")
    ax.set_xlabel("log10(frequency f / Hz)")
    ax.set_ylabel("strength g")
    ax.grid(alpha=0.25, linestyle="--")
    for r, x, y, c in zip(rows, xs, ys, cs):
        ax.text(
            x,
            y + 0.03,
            f"{c:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("improve(LER~) (%)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_phase3_bars(rows: List[Dict[str, Any]], out_path: Path) -> None:
    rows_sorted = sorted(rows, key=lambda r: float(r["improve_mean"]), reverse=True)
    labels = [f"f={_fmt_freq(float(r['f_hz']))}, g={float(r['g']):.1f}" for r in rows_sorted]
    x = np.arange(len(rows_sorted), dtype=float)
    imp = 100.0 * np.asarray([float(r["improve_mean"]) for r in rows_sorted], dtype=float)
    imp_err = 100.0 * np.asarray([float(r["improve_std"]) for r in rows_sorted], dtype=float)
    dlt = 100.0 * np.asarray([float(r["delta_mean"]) for r in rows_sorted], dtype=float)
    dlt_err = 100.0 * np.asarray([float(r["delta_std"]) for r in rows_sorted], dtype=float)
    pos = 100.0 * np.asarray([float(r["pos_ratio"]) for r in rows_sorted], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)

    axes[0].bar(x, imp, color="#2a9d8f", alpha=0.88)
    axes[0].errorbar(x, imp, yerr=imp_err, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_title("Phase3 Confirm: improve(LER~)")
    axes[0].set_ylabel("improvement (%)")

    axes[1].bar(x, dlt, color="#457b9d", alpha=0.88)
    axes[1].errorbar(x, dlt, yerr=dlt_err, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_title("Phase3 Confirm: delta success")
    axes[1].set_ylabel("percentage points")
    for i, p in enumerate(pos):
        axes[1].text(i, dlt[i] + max(0.1, 0.18 * max(1.0, np.max(dlt))), f"+seed {p:.0f}%", ha="center", fontsize=9)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot composed (f,g) RL-advantage figures.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="code/data_generated/benchmarks/composed_corr_fg",
        help="Directory containing phase*_*/summary.json outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to store generated figures. Default: <base-dir>/plots_YYYYMMDD.",
    )
    parser.add_argument("--date-tag", type=str, default="20260309")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    phase1_summary = base / "phase1_20260309" / "summary.json"
    phase2_summary = base / "phase2_20260309" / "summary.json"
    phase3_summary = base / "phase3_20260309_selected" / "summary.json"
    for p in (phase1_summary, phase2_summary, phase3_summary):
        if not p.exists():
            raise FileNotFoundError(f"Missing summary file: {p}")

    phase1_rows = _load_stage_rows(phase1_summary)
    phase2_rows = _load_stage_rows(phase2_summary)
    phase3_rows = _load_stage_rows(phase3_summary)

    out_dir = Path(args.output_dir) if str(args.output_dir).strip() else (base / f"plots_{args.date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "phase1_heatmaps_rl_advantage.png"
    p2 = out_dir / "phase2_scatter_rl_advantage.png"
    p3 = out_dir / "phase3_confirm_bars_rl_advantage.png"

    _plot_phase1_heatmaps(phase1_rows, p1)
    _plot_phase2_scatter(phase2_rows, p2)
    _plot_phase3_bars(phase3_rows, p3)

    print(f"Saved figure: {p1}")
    print(f"Saved figure: {p2}")
    print(f"Saved figure: {p3}")


if __name__ == "__main__":
    main()
