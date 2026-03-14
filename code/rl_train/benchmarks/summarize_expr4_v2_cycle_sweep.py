"""Aggregate Expr4 V2 per-seed cycle-sweep JSON files into summary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
    }


def _append_metric(bucket: dict[str, list[float]], name: str, value: float) -> None:
    bucket.setdefault(name, []).append(float(value))


def summarize_cycle_files(files: list[Path]) -> dict[str, Any]:
    if not files:
        raise ValueError("no Expr4 V2 cycle-sweep files found")

    by_round: dict[int, dict[str, dict[str, list[float]]]] = {}
    primary_label = None
    transfer_label = None

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        primary = payload["primary_policy"]
        transfer = payload.get("transfer_policy")
        fixed = payload["fixed_zero_policy"]
        primary_label = str(primary["label"])
        if transfer is not None:
            transfer_label = str(transfer["label"])

        fixed_rows = {int(row["n_rounds"]): row for row in fixed["cycle_sweep"]}
        transfer_rows = {}
        if transfer is not None:
            transfer_rows = {int(row["n_rounds"]): row for row in transfer["cycle_sweep"]}

        for row in primary["cycle_sweep"]:
            rounds = int(row["n_rounds"])
            round_bucket = by_round.setdefault(rounds, {})
            primary_bucket = round_bucket.setdefault(primary_label, {})
            fixed_bucket = round_bucket.setdefault("fixed_zero", {})

            _append_metric(primary_bucket, "success_rate", row["learned"]["success_rate"])
            _append_metric(primary_bucket, "logical_observable_proxy", row["learned"]["logical_observable_proxy"])
            _append_metric(primary_bucket, "improve_vs_fixed_zero_ler_proxy", row["improvement_vs_fixed_zero"]["ler_proxy"])

            fixed_row = fixed_rows[rounds]
            _append_metric(fixed_bucket, "success_rate", fixed_row["fixed_zero"]["success_rate"])
            _append_metric(
                fixed_bucket,
                "logical_observable_proxy",
                fixed_row["fixed_zero"]["logical_observable_proxy"],
            )

            if rounds in transfer_rows and transfer_label is not None:
                transfer_bucket = round_bucket.setdefault(transfer_label, {})
                transfer_row = transfer_rows[rounds]
                _append_metric(transfer_bucket, "success_rate", transfer_row["learned"]["success_rate"])
                _append_metric(
                    transfer_bucket,
                    "logical_observable_proxy",
                    transfer_row["learned"]["logical_observable_proxy"],
                )

    summary_rows: list[dict[str, Any]] = []
    for rounds in sorted(by_round):
        for label, metrics in sorted(by_round[rounds].items()):
            row: dict[str, Any] = {"n_rounds": int(rounds), "policy_label": label}
            for metric_name, values in sorted(metrics.items()):
                stats = _mean_std(values)
                row[f"{metric_name}_mean"] = stats["mean"]
                row[f"{metric_name}_std"] = stats["std"]
                row["n_seeds"] = stats["n"]
            summary_rows.append(row)

    return {
        "n_files": len(files),
        "primary_policy_label": primary_label,
        "transfer_policy_label": transfer_label,
        "rounds": sorted(by_round),
        "rows": summary_rows,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    rows = summary["rows"]
    labels = [summary["primary_policy_label"], summary["transfer_policy_label"], "fixed_zero"]
    lines = ["# Expr4 V2 Cycle Sweep Summary", ""]
    lines.append(f"- aggregated files: `{summary['n_files']}`")
    lines.append(f"- primary policy: `{summary['primary_policy_label']}`")
    if summary["transfer_policy_label"]:
        lines.append(f"- transfer policy: `{summary['transfer_policy_label']}`")
    lines.append("")
    lines.append("## Key Rounds")
    lines.append("")
    for rounds in (5, 25, 50):
        candidates = [r for r in rows if int(r["n_rounds"]) == rounds]
        if not candidates:
            continue
        lines.append(f"### n_rounds = {rounds}")
        for label in labels:
            if not label:
                continue
            row = next((r for r in candidates if r["policy_label"] == label), None)
            if row is None:
                continue
            lines.append(
                "- "
                f"`{label}` success={100.0 * float(row.get('success_rate_mean', float('nan'))):.2f}% "
                f"+- {100.0 * float(row.get('success_rate_std', float('nan'))):.2f}%, "
                f"logical_proxy={float(row.get('logical_observable_proxy_mean', float('nan'))):.4f} "
                f"+- {float(row.get('logical_observable_proxy_std', float('nan'))):.4f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Expr4 V2 cycle-sweep outputs.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseA_showcase_seed*.json",
        help="Glob pattern for per-seed Expr4 V2 JSON files.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/summary.md",
    )
    args = parser.parse_args()

    files = sorted(Path().glob(args.input_glob))
    summary = summarize_cycle_files(files)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown(summary), encoding="utf-8")
    print(f"Saved JSON: {output_json}")
    print(f"Saved markdown: {output_md}")


if __name__ == "__main__":
    main()
