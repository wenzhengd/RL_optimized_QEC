"""Summarize staged composed-channel (f, g) grid results into table artifacts.

Input:
  - staged_steane_experiments summary JSON.

Outputs:
  - CSV table with per-condition metrics.
  - Markdown table sorted by improve(LER~).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _mean_std(values: List[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _load_rows(summary_path: Path) -> List[Dict[str, Any]]:
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

        learned_success = [float(r["eval_metrics"]["learned"]["success_rate"]) for r in runs]
        fixed_success = [float(r["eval_metrics"]["fixed_zero"]["success_rate"]) for r in runs]
        improve_ler = [float(r["improvement_vs_fixed_zero"]["ler_proxy"]) for r in runs]
        delta_success = [l - f for l, f in zip(learned_success, fixed_success)]

        learned_mean, learned_std = _mean_std(learned_success)
        fixed_mean, fixed_std = _mean_std(fixed_success)
        improve_mean, improve_std = _mean_std(improve_ler)
        delta_mean, delta_std = _mean_std(delta_success)
        positive_ratio = float(np.mean(np.asarray(delta_success, dtype=float) > 0.0))

        trace_improve: List[float] = []
        trace_delta_success: List[float] = []
        for r in runs:
            t = r.get("trace_eval_metrics")
            if not isinstance(t, dict):
                continue
            learned_t = t.get("learned")
            fixed_t = t.get("fixed_zero")
            if not isinstance(learned_t, dict) or not isinstance(fixed_t, dict):
                continue
            ls = float(learned_t.get("success_rate", 0.0))
            fs = float(fixed_t.get("success_rate", 0.0))
            ll = float(learned_t.get("ler_proxy", 1.0 - ls))
            fl = float(fixed_t.get("ler_proxy", 1.0 - fs))
            trace_delta_success.append(ls - fs)
            trace_improve.append((fl - ll) / max(fl, 1e-12))

        if trace_improve:
            trace_improve_mean, trace_improve_std = _mean_std(trace_improve)
            trace_delta_mean, trace_delta_std = _mean_std(trace_delta_success)
            trace_pos_ratio = float(np.mean(np.asarray(trace_delta_success, dtype=float) > 0.0))
        else:
            trace_improve_mean, trace_improve_std = float("nan"), float("nan")
            trace_delta_mean, trace_delta_std = float("nan"), float("nan")
            trace_pos_ratio = float("nan")

        rows.append(
            {
                "stage_name": str(stage_name),
                "f_hz": f_hz,
                "g": g,
                "n_seeds": len(runs),
                "improve_ler_mean": improve_mean,
                "improve_ler_std": improve_std,
                "learned_success_mean": learned_mean,
                "learned_success_std": learned_std,
                "fixed_success_mean": fixed_mean,
                "fixed_success_std": fixed_std,
                "delta_success_mean": delta_mean,
                "delta_success_std": delta_std,
                "positive_seed_ratio": positive_ratio,
                "trace_improve_ler_mean": trace_improve_mean,
                "trace_improve_ler_std": trace_improve_std,
                "trace_delta_success_mean": trace_delta_mean,
                "trace_delta_success_std": trace_delta_std,
                "trace_positive_seed_ratio": trace_pos_ratio,
            }
        )

    rows.sort(key=lambda x: (x["f_hz"], x["g"]))
    return rows


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage_name",
        "f_hz",
        "g",
        "n_seeds",
        "improve_ler_mean",
        "improve_ler_std",
        "learned_success_mean",
        "learned_success_std",
        "fixed_success_mean",
        "fixed_success_std",
        "delta_success_mean",
        "delta_success_std",
        "positive_seed_ratio",
        "trace_improve_ler_mean",
        "trace_improve_ler_std",
        "trace_delta_success_mean",
        "trace_delta_success_std",
        "trace_positive_seed_ratio",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_pct(x: float) -> str:
    if not np.isfinite(float(x)):
        return "-"
    return f"{100.0 * x:+.2f}%"


def _write_markdown(rows: List[Dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    by_improve = sorted(rows, key=lambda x: x["improve_ler_mean"], reverse=True)
    lines: List[str] = []
    lines.append(
        "| stage | f(Hz) | g | seeds | fast improve(LER~) | fast delta success | fast +seed | "
        "trace improve(LER~) | trace delta success | trace +seed | learned success | fixed success |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in by_improve:
        trace_pos = row["trace_positive_seed_ratio"]
        trace_pos_txt = "-" if not np.isfinite(float(trace_pos)) else f"{100.0 * trace_pos:.1f}%"
        lines.append(
            "| "
            + f"{row['stage_name']} | "
            + f"{row['f_hz']:.3g} | "
            + f"{row['g']:.3g} | "
            + f"{row['n_seeds']} | "
            + f"{_fmt_pct(row['improve_ler_mean'])} +- {100.0 * row['improve_ler_std']:.2f}% | "
            + f"{_fmt_pct(row['delta_success_mean'])} +- {100.0 * row['delta_success_std']:.2f}% | "
            + f"{100.0 * row['positive_seed_ratio']:.1f}% | "
            + f"{_fmt_pct(row['trace_improve_ler_mean'])} +- "
            + (
                "-"
                if not np.isfinite(float(row["trace_improve_ler_std"]))
                else f"{100.0 * row['trace_improve_ler_std']:.2f}%"
            )
            + " | "
            + f"{_fmt_pct(row['trace_delta_success_mean'])} +- "
            + (
                "-"
                if not np.isfinite(float(row["trace_delta_success_std"]))
                else f"{100.0 * row['trace_delta_success_std']:.2f}%"
            )
            + " | "
            + f"{trace_pos_txt} | "
            + f"{100.0 * row['learned_success_mean']:.2f}% +- {100.0 * row['learned_success_std']:.2f}% | "
            + f"{100.0 * row['fixed_success_mean']:.2f}% +- {100.0 * row['fixed_success_std']:.2f}% |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize composed-channel (f,g) stage summary.")
    parser.add_argument("--summary-json", type=str, required=True, help="Path to staged summary JSON.")
    parser.add_argument("--output-csv", type=str, default="", help="Optional output CSV path.")
    parser.add_argument("--output-md", type=str, default="", help="Optional output markdown path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    rows = _load_rows(summary_path)
    if not rows:
        raise RuntimeError(f"No stage rows found in {summary_path}")

    default_base = summary_path.parent / "composed_fg_grid_summary"
    out_csv = Path(args.output_csv) if str(args.output_csv).strip() else default_base.with_suffix(".csv")
    out_md = Path(args.output_md) if str(args.output_md).strip() else default_base.with_suffix(".md")

    _write_csv(rows, out_csv)
    _write_markdown(rows, out_md)

    best = max(rows, key=lambda x: x["improve_ler_mean"])
    print(f"Saved CSV: {out_csv}")
    print(f"Saved markdown: {out_md}")
    print(
        "Best by improve(LER~): "
        f"{best['stage_name']} (f={best['f_hz']:.3g}, g={best['g']:.3g}, "
        f"improve={100.0 * best['improve_ler_mean']:.2f}%)"
    )


if __name__ == "__main__":
    main()
