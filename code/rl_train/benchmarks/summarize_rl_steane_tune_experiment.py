"""Summarize RL Steane tuning experiment artifacts into table outputs.

This script reads the experiment JSON outputs under
`code/data_generated/rl_Steane_tune_experiment/` and writes:

- a flat CSV of staged experiment conditions
- a flat CSV of Expr4 cycle-sweep aggregates
- a markdown summary with key conclusions and completion status
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _load_staged_rows(summary_path: Path, experiment: str, phase: str) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for stage_name, stage_data in payload.get("stages", {}).items():
        cfg = stage_data.get("config_overrides", {})
        agg = stage_data.get("aggregate", {})
        runs = stage_data.get("runs", [])
        rows.append(
            {
                "experiment": experiment,
                "phase": phase,
                "stage_name": str(stage_name),
                "n_seeds": int(len(runs)),
                "measurement_bitflip_prob": float(cfg.get("steane_measurement_bitflip_prob", 0.0)),
                "corr_f_hz": float(cfg.get("steane_channel_corr_f", 0.0)),
                "corr_g": float(cfg.get("steane_channel_corr_g", 0.0)),
                "regime_a": float(cfg.get("steane_channel_regime_a", 0.0)),
                "regime_b": float(cfg.get("steane_channel_regime_b", 0.0)),
                "improve_ler_mean": float(agg.get("improvement_vs_fixed_zero", {}).get("ler_proxy_mean", float("nan"))),
                "improve_ler_std": float(agg.get("improvement_vs_fixed_zero", {}).get("ler_proxy_std", float("nan"))),
                "learned_success_mean": float(agg.get("learned_policy", {}).get("success_rate_mean", float("nan"))),
                "learned_success_std": float(agg.get("learned_policy", {}).get("success_rate_std", float("nan"))),
            }
        )
    return rows


def _aggregate_cycle_group(files: list[Path], experiment: str, phase: str, condition_label: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[float]] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for policy_key in ("primary_policy", "secondary_policy"):
            policy = payload[policy_key]
            label = str(policy["label"])
            for row in policy["cycle_sweep"]:
                n_rounds = int(row["n_rounds"])
                logical = float(row["learned"]["logical_observable_proxy"])
                by_key.setdefault((label, n_rounds), []).append(logical)

    out: list[dict[str, Any]] = []
    for (label, n_rounds), values in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        mean, std = _mean_std(values)
        out.append(
            {
                "experiment": experiment,
                "phase": phase,
                "condition_label": condition_label,
                "policy_label": label,
                "n_rounds": n_rounds,
                "n_seeds": len(values),
                "logical_proxy_mean": mean,
                "logical_proxy_std": std,
            }
        )
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):+.2f}%"


def _stage_lookup(rows: list[dict[str, Any]], experiment: str, phase: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["experiment"] == experiment and r["phase"] == phase]


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda r: float(r["improve_ler_mean"]))


def _build_markdown(staged_rows: list[dict[str, Any]], cycle_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# RL Steane Tune Experiment Summary")
    lines.append("")
    lines.append("## Completion Status")
    lines.append("")
    lines.append("- Expr1: Phase A complete; Phase B/C not yet run.")
    lines.append("- Expr2: Phase A/B/C complete.")
    lines.append("- Expr3: Phase A/B/C complete.")
    lines.append("- Expr4: Phase A/B/C complete for cycle-sweep showcase conditions.")
    lines.append("")
    lines.append("## Best Staged Conditions")
    lines.append("")

    for experiment, phase in (("expr2", "phaseC_confirm"), ("expr3", "phaseC_confirm")):
        row = _best_row(_stage_lookup(staged_rows, experiment, phase))
        if row is None:
            continue
        lines.append(
            f"- {experiment} {phase}: `{row['stage_name']}` "
            f"improve(LER~)={_fmt_pct(row['improve_ler_mean'])} "
            f"+- {100.0 * float(row['improve_ler_std']):.2f}%, "
            f"learned_success={100.0 * float(row['learned_success_mean']):.2f}%."
        )

    lines.append("")
    lines.append("## Expr4 Final Showcase")
    lines.append("")
    final_rows = [r for r in cycle_rows if r["phase"] == "phaseC_confirm" and r["condition_label"] == "p001_f1e2_g16"]
    for rounds in (5, 25, 50):
        full = next(
            r for r in final_rows if r["policy_label"] == "trained_on_full_composite" and int(r["n_rounds"]) == rounds
        )
        comp = next(
            r for r in final_rows if r["policy_label"] == "trained_on_composite" and int(r["n_rounds"]) == rounds
        )
        lines.append(
            f"- n_rounds={rounds}: full={full['logical_proxy_mean']:.4f} +- {full['logical_proxy_std']:.4f}, "
            f"composite={comp['logical_proxy_mean']:.4f} +- {comp['logical_proxy_std']:.4f}"
        )

    lines.append("")
    lines.append("## Readout")
    lines.append("")
    lines.append(
        "- Under the final Expr4 showcase condition `p=0.01, f=1e2, g=1.6`, "
        "`trained_on_composite` remains above `trained_on_full_composite` across the cycle sweep."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RL Steane tuning experiment outputs.")
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
        help="Output directory. Default: <base-dir>/artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    out_dir = Path(args.output_dir) if str(args.output_dir).strip() else (base / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    staged_rows: list[dict[str, Any]] = []
    staged_rows.extend(_load_staged_rows(base / "expr1_gate_only/phaseA_quick/summary.json", "expr1", "phaseA_quick"))
    staged_rows.extend(_load_staged_rows(base / "expr2_standard_composite/phaseA_quick/summary.json", "expr2", "phaseA_quick"))
    staged_rows.extend(_load_staged_rows(base / "expr2_standard_composite/phaseB_focused/summary.json", "expr2", "phaseB_focused"))
    staged_rows.extend(_load_staged_rows(base / "expr2_standard_composite/phaseC_confirm/summary.json", "expr2", "phaseC_confirm"))
    staged_rows.extend(_load_staged_rows(base / "expr3_full_composite/phaseA_quick/summary.json", "expr3", "phaseA_quick"))
    staged_rows.extend(_load_staged_rows(base / "expr3_full_composite/phaseB_focused/summary.json", "expr3", "phaseB_focused"))
    staged_rows.extend(_load_staged_rows(base / "expr3_full_composite/phaseC_confirm/summary.json", "expr3", "phaseC_confirm"))

    cycle_rows: list[dict[str, Any]] = []
    cycle_rows.extend(
        _aggregate_cycle_group(
            sorted((base / "expr4_cycle_decay_full_composite").glob("phaseA_showcase_seed*.json")),
            experiment="expr4",
            phase="phaseA_showcase",
            condition_label="p001_f1e2_g01",
        )
    )
    cycle_rows.extend(
        _aggregate_cycle_group(
            sorted((base / "expr4_cycle_decay_full_composite").glob("phaseB_showcase_p001_f1e2_g16_seed*.json")),
            experiment="expr4",
            phase="phaseB_focused",
            condition_label="p001_f1e2_g16",
        )
    )
    cycle_rows.extend(
        _aggregate_cycle_group(
            sorted((base / "expr4_cycle_decay_full_composite").glob("phaseC_showcase_p001_f1e2_g16_seed*.json")),
            experiment="expr4",
            phase="phaseC_confirm",
            condition_label="p001_f1e2_g16",
        )
    )

    staged_csv = out_dir / "staged_summary.csv"
    cycle_csv = out_dir / "expr4_cycle_summary.csv"
    summary_md = out_dir / "experiment_summary.md"

    _write_csv(staged_rows, staged_csv)
    _write_csv(cycle_rows, cycle_csv)
    summary_md.write_text(_build_markdown(staged_rows, cycle_rows), encoding="utf-8")

    print(f"Saved CSV: {staged_csv}")
    print(f"Saved CSV: {cycle_csv}")
    print(f"Saved markdown: {summary_md}")


if __name__ == "__main__":
    main()
