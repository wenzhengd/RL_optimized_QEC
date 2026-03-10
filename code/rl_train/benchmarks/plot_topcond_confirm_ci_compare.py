"""Plot 95% CI comparison for top-condition confirm runs (10 vs 40 seeds)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _short_label(stage_name: str) -> str:
    s = stage_name.lower()
    if "longtrace" in s:
        return "Long-Trace"
    if "baseline" in s:
        return "Baseline"
    return stage_name


def _ci95_halfwidth(std: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return 1.96 * float(std) / np.sqrt(float(n))


def _load_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Empty CSV: {csv_path}")
    df = df.copy()
    df["label"] = df["stage_name"].map(_short_label)
    return df


def plot_ci_compare(df: pd.DataFrame, out_png: Path) -> None:
    labels = ["Baseline", "Long-Trace"]
    ns = sorted(df["n_seeds"].astype(int).unique().tolist())
    if ns != [10, 40]:
        # Keep plotting generic, but title still mentions confirm comparison.
        pass

    colors = {"Baseline": "#4e79a7", "Long-Trace": "#f28e2b"}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)

    for label in labels:
        sub = df[df["label"] == label].sort_values("n_seeds")
        x = sub["n_seeds"].astype(int).to_numpy()

        fast_mean = 100.0 * sub["improve_ler_mean"].to_numpy(dtype=float)
        fast_ci = 100.0 * np.asarray(
            [_ci95_halfwidth(s, int(n)) for s, n in zip(sub["improve_ler_std"], sub["n_seeds"])],
            dtype=float,
        )

        trace_mean = 100.0 * sub["trace_improve_ler_mean"].to_numpy(dtype=float)
        trace_ci = 100.0 * np.asarray(
            [_ci95_halfwidth(s, int(n)) for s, n in zip(sub["trace_improve_ler_std"], sub["n_seeds"])],
            dtype=float,
        )

        axes[0].errorbar(
            x,
            fast_mean,
            yerr=fast_ci,
            fmt="o-",
            capsize=4,
            linewidth=1.8,
            color=colors[label],
            label=label,
        )
        axes[1].errorbar(
            x,
            trace_mean,
            yerr=trace_ci,
            fmt="o-",
            capsize=4,
            linewidth=1.8,
            color=colors[label],
            label=label,
        )

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Fast eval: improve(LER~) with 95% CI")
    axes[0].set_xlabel("n_seeds")
    axes[0].set_ylabel("improve(LER~) (%)")
    axes[0].set_xticks(ns)

    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Trace eval: improve(LER~) with 95% CI")
    axes[1].set_xlabel("n_seeds")
    axes[1].set_ylabel("improve(LER~) (%)")
    axes[1].set_xticks(ns)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 95% CI comparison for confirm10 vs confirm40.")
    parser.add_argument("--confirm10-csv", type=str, required=True)
    parser.add_argument("--confirm40-csv", type=str, required=True)
    parser.add_argument("--output-png", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c10 = _load_rows(Path(args.confirm10_csv))
    c40 = _load_rows(Path(args.confirm40_csv))
    df = pd.concat([c10, c40], ignore_index=True)
    plot_ci_compare(df, Path(args.output_png))
    print(f"Saved figure: {args.output_png}")


if __name__ == "__main__":
    main()
