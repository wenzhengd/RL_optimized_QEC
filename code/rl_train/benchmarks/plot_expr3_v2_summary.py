"""Plot Expr3 V2 summary figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Expr3 V2 summary figures.")
    parser.add_argument("--base-dir", type=str, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_stage_cfg(row: dict) -> tuple[float, float, float, float]:
    cfg = row["config_overrides"]
    return (
        float(cfg["steane_channel_regime_a"]),
        float(cfg["steane_channel_corr_f"]),
        float(cfg["steane_channel_corr_g"]),
        float(cfg["steane_measurement_bitflip_prob"]),
    )


def plot_phasea(phasea: dict, out_path: Path) -> None:
    rows = []
    for name, row in phasea["stages"].items():
        scale, f_hz, g, p_meas = _parse_stage_cfg(row)
        rows.append(
            {
                "name": name,
                "scale": scale,
                "f_hz": f_hz,
                "g": g,
                "p_meas": p_meas,
                "imp": float(row["aggregate"]["improvement_vs_fixed_zero"]["ler_proxy_mean"]),
                "imp_std": float(row["aggregate"]["improvement_vs_fixed_zero"]["ler_proxy_std"]),
            }
        )

    anchor_order = [
        (0.025, 1.0e4, 1.0),
        (0.025, 1.0e3, 1.6),
        (0.02, 1.0e2, 0.4),
    ]
    p_order = [0.001, 0.003, 0.01]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), sharey=True, constrained_layout=True)
    for ax, anchor in zip(axes, anchor_order):
        vals = []
        errs = []
        labels = []
        for p in p_order:
            row = next(r for r in rows if (r["scale"], r["f_hz"], r["g"], r["p_meas"]) == (*anchor, p))
            vals.append(100.0 * row["imp"])
            errs.append(100.0 * row["imp_std"])
            labels.append(f"{p:g}")
        x = np.arange(len(p_order), dtype=float)
        ax.bar(x, vals, color="#2a9d8f", width=0.66)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xticks(x, labels)
        ax.set_xlabel("p_meas")
        ax.set_title(f"s={anchor[0]:g}, f={anchor[1]:.0f}, g={anchor[2]:.1f}")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        for xpos, val in zip(x, vals):
            ax.text(xpos, val + 1.0, f"{val:+.1f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("improve(LER~) (%)")
    fig.suptitle("Expr3 V2 Phase A: Measurement-Noise Sweep")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_phasebc(phaseb: dict, phasec: dict, out_path: Path) -> None:
    rows = []
    for phase_name, payload in (("Phase B", phaseb), ("Phase C", phasec)):
        for _, row in payload["stages"].items():
            scale, f_hz, g, p_meas = _parse_stage_cfg(row)
            rows.append(
                {
                    "phase": phase_name,
                    "label": f"s={scale:g}, f={f_hz:.0f}, g={g:.1f}, p={p_meas:g}",
                    "imp": 100.0 * float(row["aggregate"]["improvement_vs_fixed_zero"]["ler_proxy_mean"]),
                    "imp_std": 100.0 * float(row["aggregate"]["improvement_vs_fixed_zero"]["ler_proxy_std"]),
                }
            )

    order = [
        "s=0.025, f=10000, g=1.0, p=0.003",
        "s=0.025, f=1000, g=1.6, p=0.01",
        "s=0.025, f=10000, g=1.0, p=0.01",
    ]
    x = np.arange(len(order), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    phase_to_offset = {"Phase B": -width / 2.0, "Phase C": width / 2.0}
    phase_to_color = {"Phase B": "#457b9d", "Phase C": "#2a9d8f"}

    for phase in ("Phase B", "Phase C"):
        vals = [next(r["imp"] for r in rows if r["phase"] == phase and r["label"] == label) for label in order]
        errs = [next(r["imp_std"] for r in rows if r["phase"] == phase and r["label"] == label) for label in order]
        xpos = x + phase_to_offset[phase]
        ax.bar(xpos, vals, width=width, color=phase_to_color[phase], label=phase)
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
        for xx, val in zip(xpos, vals):
            ax.text(xx, val + 0.9, f"{val:+.1f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x, order, rotation=18, ha="right")
    ax.set_ylabel("improve(LER~) (%)")
    ax.set_title("Expr3 V2 Phase B/C Comparison")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    phasea = _load(base / "phaseA_pilot" / "summary.json")
    phaseb = _load(base / "phaseB_focused" / "summary.json")
    phasec = _load(base / "phaseC_confirm" / "summary.json")
    out_dir = base / "plots"
    plot_phasea(phasea, out_dir / "expr3_v2_phaseA_p_sweep.png")
    plot_phasebc(phaseb, phasec, out_dir / "expr3_v2_phaseBC_compare.png")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
