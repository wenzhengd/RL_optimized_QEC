"""Sweep Steane channel-regime parameters with a fixed RL configuration.

This script reuses `eval_steane_ppo` so we do not duplicate training/eval logic.
Use it to compare RL performance against channel parameter regimes `(a, b)`.
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
    parser = argparse.ArgumentParser(description="Sweep Steane channel regime parameters (a,b).")
    parser.add_argument("--regime-a-values", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--regime-b-values", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--force-channel", type=str, default="parametric_google")
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

    regime_a_values = _parse_float_list(sweep_args.regime_a_values)
    regime_b_values = _parse_float_list(sweep_args.regime_b_values)

    runs: list[dict[str, Any]] = []
    for a in regime_a_values:
        for b in regime_b_values:
            run_args = copy.deepcopy(base_eval_args)
            run_args.steane_noise_channel = str(sweep_args.force_channel)
            run_args.steane_channel_regime_a = float(a)
            run_args.steane_channel_regime_b = float(b)
            report = run_benchmark(run_args)
            runs.append(
                {
                    "regime_a": float(a),
                    "regime_b": float(b),
                    "improvement_vs_fixed_zero": report["improvement_vs_fixed_zero"],
                    "eval_metrics": report["eval_metrics"],
                    "phase1_mean_rollout_reward": report.get("phase1_mean_rollout_reward"),
                    "trace_finetune_enabled": report.get("trace_finetune_enabled", False),
                    "trace_finetune_mean_rollout_reward": report.get("trace_finetune_mean_rollout_reward"),
                }
            )
            print(
                f"a={a:.4g}, b={b:.4g}: "
                f"improve(LER~)={100.0 * report['improvement_vs_fixed_zero']['ler_proxy']:+.2f}%, "
                f"learned_success={100.0 * report['eval_metrics']['learned']['success_rate']:.2f}%"
            )

    payload = {
        "grid": {
            "regime_a_values": regime_a_values,
            "regime_b_values": regime_b_values,
            "force_channel": str(sweep_args.force_channel),
        },
        "base_eval_args": vars(base_eval_args),
        "aggregate": _aggregate_grid(runs),
        "runs": runs,
    }

    out_path = Path(sweep_args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved regime sweep report: {out_path}")


if __name__ == "__main__":
    main()
