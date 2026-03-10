"""Plot top-condition ablation results (baseline vs tune-a vs long-trace)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _mean_std(vals: List[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _extract_rows(summary_json: Path) -> List[Dict[str, float | str]]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    rows: List[Dict[str, float | str]] = []
    for stage_name, stage_data in payload.get("stages", {}).items():
        runs = stage_data.get("runs", [])
        if not runs:
            continue

        fast_imp: List[float] = []
        fast_delta: List[float] = []
        trace_imp: List[float] = []
        trace_delta: List[float] = []

        for r in runs:
            el = float(r["eval_metrics"]["learned"]["ler_proxy"])
            ef = float(r["eval_metrics"]["fixed_zero"]["ler_proxy"])
            es = float(r["eval_metrics"]["learned"]["success_rate"])
            fs = float(r["eval_metrics"]["fixed_zero"]["success_rate"])
            fast_imp.append((ef - el) / max(ef, 1e-12))
            fast_delta.append(es - fs)

            t = r.get("trace_eval_metrics")
            if isinstance(t, dict):
                tl = float(t["learned"]["ler_proxy"])
                tf = float(t["fixed_zero"]["ler_proxy"])
                ts = float(t["learned"]["success_rate"])
                tfs = float(t["fixed_zero"]["success_rate"])
                trace_imp.append((tf - tl) / max(tf, 1e-12))
                trace_delta.append(ts - tfs)

        fim, fis = _mean_std(fast_imp)
        fdm, fds = _mean_std(fast_delta)
        tim, tis = (_mean_std(trace_imp) if trace_imp else (float("nan"), float("nan")))
        tdm, tds = (_mean_std(trace_delta) if trace_delta else (float("nan"), float("nan")))

        rows.append(
            {
                "stage_name": str(stage_name),
                "fast_improve_mean": fim,
                "fast_improve_std": fis,
                "fast_delta_mean": fdm,
                "fast_delta_std": fds,
                "trace_improve_mean": tim,
                "trace_improve_std": tis,
                "trace_delta_mean": tdm,
                "trace_delta_std": tds,
            }
        )
    return rows


def _short_label(stage_name: str) -> str:
    s = stage_name.lower()
    if "tunea" in s:
        return "Tune-A"
    if "longtrace" in s:
        return "Long-Trace"
    if "baseline" in s:
        return "Baseline"
    return stage_name


def _sort_rows(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    order = {"Baseline": 0, "Tune-A": 1, "Long-Trace": 2}
    return sorted(rows, key=lambda r: order.get(_short_label(str(r["stage_name"])), 99))


def plot_ablation(rows: List[Dict[str, float | str]], out_png: Path) -> None:
    rows = _sort_rows(rows)
    labels = [_short_label(str(r["stage_name"])) for r in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.38

    fim = 100.0 * np.asarray([float(r["fast_improve_mean"]) for r in rows], dtype=float)
    fis = 100.0 * np.asarray([float(r["fast_improve_std"]) for r in rows], dtype=float)
    tim = 100.0 * np.asarray([float(r["trace_improve_mean"]) for r in rows], dtype=float)
    tis = 100.0 * np.asarray([float(r["trace_improve_std"]) for r in rows], dtype=float)

    fdm = 100.0 * np.asarray([float(r["fast_delta_mean"]) for r in rows], dtype=float)
    fds = 100.0 * np.asarray([float(r["fast_delta_std"]) for r in rows], dtype=float)
    tdm = 100.0 * np.asarray([float(r["trace_delta_mean"]) for r in rows], dtype=float)
    tds = 100.0 * np.asarray([float(r["trace_delta_std"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)

    axes[0].bar(x - width / 2, fim, width=width, color="#2a9d8f", label="fast eval")
    axes[0].errorbar(x - width / 2, fim, yerr=fis, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    axes[0].bar(x + width / 2, tim, width=width, color="#e76f51", label="trace eval")
    axes[0].errorbar(x + width / 2, tim, yerr=tis, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("improve(LER~) (%)")
    axes[0].set_title("Top Condition Ablation: LER Improvement")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, fdm, width=width, color="#457b9d", label="fast eval")
    axes[1].errorbar(x - width / 2, fdm, yerr=fds, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    axes[1].bar(x + width / 2, tdm, width=width, color="#f4a261", label="trace eval")
    axes[1].errorbar(x + width / 2, tdm, yerr=tds, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("delta success (percentage points)")
    axes[1].set_title("Top Condition Ablation: Success Gain")
    axes[1].legend(frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top-condition ablation charts.")
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-png", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = Path(args.summary_json)
    rows = _extract_rows(summary)
    if not rows:
        raise RuntimeError(f"No rows in summary: {summary}")
    out_png = (
        Path(args.output_png)
        if str(args.output_png).strip()
        else summary.parent / "topcond_ablation_fast_vs_trace.png"
    )
    plot_ablation(rows, out_png)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
