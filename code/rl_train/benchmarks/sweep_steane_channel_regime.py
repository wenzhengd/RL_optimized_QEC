"""Sweep correlated-channel parameters (f, g) with a fixed RL configuration.

This script reuses `eval_steane_ppo` so we do not duplicate training/eval logic.
Use it to compare RL performance against correlated-channel regimes `(f, g)`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .eval_steane_ppo import parse_args as parse_eval_args
from .eval_steane_ppo import run_benchmark


def _parse_float_list(text: str) -> list[float]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return [float(v) for v in vals]


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse sweep-specific args and leave the rest for eval_steane_ppo."""
    parser = argparse.ArgumentParser(description="Sweep Steane correlated-channel parameters (f,g).")
    parser.add_argument("--corr-f-values", type=str, default="")
    parser.add_argument("--corr-g-values", type=str, default="")
    # Deprecated aliases.
    parser.add_argument("--regime-a-values", type=str, default="0.6,1.0,1.4")
    parser.add_argument("--regime-b-values", type=str, default="1e3,1e4,1e5")
    parser.add_argument("--force-channel", type=str, default="correlated_pauli_noise_channel")
    parser.add_argument("--output-json", type=str, default="code/data_generated/steane_channel_regime_sweep.json")
    return parser.parse_known_args(argv)


def _aggregate_grid(reports: list[dict[str, Any]]) -> dict[str, float]:
    ler_imps = np.asarray([r["improvement_vs_fixed_zero"]["ler_proxy"] for r in reports], dtype=float)
    succ = np.asarray([r["eval_metrics"]["learned"]["success_rate"] for r in reports], dtype=float)
    return {
        "improve_ler_proxy_mean": float(np.mean(ler_imps)),
        "improve_ler_proxy_std": float(np.std(ler_imps)),
        "learned_success_mean": float(np.mean(succ)),
        "learned_success_std": float(np.std(succ)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    sweep_args, rest = parse_args(argv)
    base_eval_args = parse_eval_args(rest)

    corr_f_values = (
        _parse_float_list(sweep_args.corr_f_values)
        if str(sweep_args.corr_f_values).strip()
        else _parse_float_list(sweep_args.regime_b_values)
    )
    corr_g_values = (
        _parse_float_list(sweep_args.corr_g_values)
        if str(sweep_args.corr_g_values).strip()
        else _parse_float_list(sweep_args.regime_a_values)
    )

    runs: list[dict[str, Any]] = []
    for f_hz in corr_f_values:
        for g in corr_g_values:
            run_args = copy.deepcopy(base_eval_args)
            run_args.steane_noise_channel = str(sweep_args.force_channel)
            run_args.steane_channel_corr_f = float(f_hz)
            run_args.steane_channel_corr_g = float(g)
            report = run_benchmark(run_args)
            runs.append(
                {
                    "corr_f_hz": float(f_hz),
                    "corr_g": float(g),
                    "improvement_vs_fixed_zero": report["improvement_vs_fixed_zero"],
                    "eval_metrics": report["eval_metrics"],
                    "phase1_mean_rollout_reward": report.get("phase1_mean_rollout_reward"),
                    "trace_finetune_enabled": report.get("trace_finetune_enabled", False),
                    "trace_finetune_mean_rollout_reward": report.get("trace_finetune_mean_rollout_reward"),
                }
            )
            print(
                f"f={f_hz:.4g}Hz, g={g:.4g}: "
                f"improve(LER~)={100.0 * report['improvement_vs_fixed_zero']['ler_proxy']:+.2f}%, "
                f"learned_success={100.0 * report['eval_metrics']['learned']['success_rate']:.2f}%"
            )

    payload = {
        "grid": {
            "corr_f_values": corr_f_values,
            "corr_g_values": corr_g_values,
            "force_channel": str(sweep_args.force_channel),
        },
        "base_eval_args": vars(base_eval_args),
        "aggregate": _aggregate_grid(runs),
        "runs": runs,
    }

    out_path = Path(sweep_args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved correlated-channel sweep report: {out_path}")


if __name__ == "__main__":
    main()
