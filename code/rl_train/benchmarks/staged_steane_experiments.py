"""Three-stage Steane PPO experiment driver.

Stages follow a practical progression:
  1) sanity: tiny compute budget, verify pipeline runs end-to-end
  2) pilot: moderate budget, check whether RL shows positive signal
  3) scale: larger budget, probe improvement under heavier training/eval
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .eval_steane_ppo import parse_args as parse_eval_args
from .eval_steane_ppo import print_report, run_benchmark


@dataclass(frozen=True)
class StageSpec:
    """Configuration block for one experiment stage."""

    name: str
    description: str
    seed_list: List[int]
    overrides: Dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI for staged experiment orchestration."""
    parser = argparse.ArgumentParser(description="Run staged Steane PPO experiments.")
    parser.add_argument(
        "--stages",
        type=str,
        default="1,2,3",
        help="Comma-separated stage IDs to run from {1,2,3,4}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="code/data_generated/steane_staged_runs",
        help="Directory to store per-run and summary JSON reports.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--print-each-run",
        action="store_true",
        help="Print full per-run report blocks in addition to compact summary.",
    )
    return parser.parse_args(argv)


def _default_stage_specs(seed_offset: int, device: str) -> Dict[str, StageSpec]:
    """Build canonical stage specs with progressively larger compute budgets."""
    return {
        "1": StageSpec(
            name="stage1_sanity",
            description="Tiny run to verify code path and metrics plumbing.",
            seed_list=[42 + seed_offset],
            overrides={
                # Keep False here: this staged driver sets explicit overrides
                # and must avoid later preset re-overriding them.
                "google_paper_ppo_preset": False,
                "device": device,
                # Keep stage-1 intentionally tiny so users can quickly verify
                # end-to-end execution on a laptop.
                "total_timesteps": 20,
                "rollout_steps": 2,
                "steane_n_rounds": 1,
                "steane_shots_per_step": 1,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
                "post_eval_episodes": 2,
            },
        ),
        "2": StageSpec(
            name="stage2_pilot",
            description="Moderate run to test whether learned policy beats baselines.",
            seed_list=[42 + seed_offset, 43 + seed_offset],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Scaled to be ~10x total compute of stage1 (with 2 seeds).
                "total_timesteps": 60,
                "rollout_steps": 10,
                "steane_n_rounds": 1,
                "steane_shots_per_step": 2,
                # Thread workers may not speed up Python-heavy loops due to GIL.
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
                "post_eval_episodes": 2,
            },
        ),
        "3": StageSpec(
            name="stage3_scale",
            description="Larger run for performance scaling within workstation limits.",
            seed_list=[45 + seed_offset, 46 + seed_offset, 47 + seed_offset],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Scaled to be ~10x total compute of stage2 (with 3 seeds).
                "total_timesteps": 210,
                "rollout_steps": 16,
                "steane_n_rounds": 2,
                "steane_shots_per_step": 2,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
                "post_eval_episodes": 2,
            },
        ),
        "4": StageSpec(
            name="stage4_power",
            description=(
                "Power-focused run: larger seeds/eval budget with higher-evidence "
                "comparison while keeping workstation runtime practical."
            ),
            seed_list=[50 + seed_offset, 51 + seed_offset, 52 + seed_offset, 53 + seed_offset, 54 + seed_offset],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Keep training modest but non-trivial.
                "total_timesteps": 512,
                "rollout_steps": 32,
                "steane_n_rounds": 4,
                "steane_shots_per_step": 4,
                # Increase post-train statistics without inflating training cost.
                "post_eval_episodes": 8,
                "eval_steane_shots_per_step": 24,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
    }


def _apply_overrides(base, overrides: Dict[str, Any]):
    """Mutate argparse namespace with stage-specific overrides."""
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _aggregate_stage_metrics(run_reports: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std summaries for core metrics across seeds."""
    dr_imp = np.asarray([r["improvement_vs_fixed_zero"]["detector_rate"] for r in run_reports], dtype=float)
    ler_imp = np.asarray([r["improvement_vs_fixed_zero"]["ler_proxy"] for r in run_reports], dtype=float)
    succ = np.asarray([r["eval_metrics"]["learned"]["success_rate"] for r in run_reports], dtype=float)
    ler = np.asarray([r["eval_metrics"]["learned"]["ler_proxy"] for r in run_reports], dtype=float)
    return {
        "improvement_vs_fixed_zero": {
            "detector_rate_mean": float(np.mean(dr_imp)),
            "detector_rate_std": float(np.std(dr_imp)),
            "ler_proxy_mean": float(np.mean(ler_imp)),
            "ler_proxy_std": float(np.std(ler_imp)),
        },
        "learned_policy": {
            "success_rate_mean": float(np.mean(succ)),
            "success_rate_std": float(np.std(succ)),
            "ler_proxy_mean": float(np.mean(ler)),
            "ler_proxy_std": float(np.std(ler)),
        },
    }


def _compact_stage_line(stage_name: str, agg: Dict[str, Dict[str, float]]) -> str:
    """Single-line summary to quickly compare stage outcomes."""
    dr_m = 100.0 * agg["improvement_vs_fixed_zero"]["detector_rate_mean"]
    dr_s = 100.0 * agg["improvement_vs_fixed_zero"]["detector_rate_std"]
    ler_m = 100.0 * agg["improvement_vs_fixed_zero"]["ler_proxy_mean"]
    ler_s = 100.0 * agg["improvement_vs_fixed_zero"]["ler_proxy_std"]
    succ_m = 100.0 * agg["learned_policy"]["success_rate_mean"]
    succ_s = 100.0 * agg["learned_policy"]["success_rate_std"]
    return (
        f"{stage_name}: "
        f"improve(DR)={dr_m:+.2f}%±{dr_s:.2f}, "
        f"improve(LER~)={ler_m:+.2f}%±{ler_s:.2f}, "
        f"learned_success={succ_m:.2f}%±{succ_s:.2f}"
    )


def main() -> None:
    args = parse_args()
    stage_order = [s.strip() for s in args.stages.split(",") if s.strip()]
    stage_specs = _default_stage_specs(seed_offset=args.seed_offset, device=args.device)
    unknown = [s for s in stage_order if s not in stage_specs]
    if unknown:
        raise ValueError(f"Unknown stage ids: {unknown}. Allowed: 1,2,3,4.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stage_reports: Dict[str, Any] = {"stages": {}}
    for sid in stage_order:
        spec = stage_specs[sid]
        print(f"\n=== {spec.name} ===")
        print(spec.description)

        run_reports: List[Dict[str, Any]] = []
        stage_dir = out_dir / spec.name
        stage_dir.mkdir(parents=True, exist_ok=True)

        for seed in spec.seed_list:
            # Start from eval script defaults to keep behavior aligned.
            run_args = parse_eval_args([])
            run_args = _apply_overrides(run_args, spec.overrides)
            run_args.seed = int(seed)
            run_args.save_json = str(stage_dir / f"seed_{seed}.json")

            report = run_benchmark(run_args)
            run_reports.append(report)
            print(
                f"seed={seed}: "
                f"learned_success={100.0 * report['eval_metrics']['learned']['success_rate']:.2f}%, "
                f"improve(LER~)={100.0 * report['improvement_vs_fixed_zero']['ler_proxy']:+.2f}%"
            )
            if args.print_each_run:
                print_report(report)

        agg = _aggregate_stage_metrics(run_reports)
        print(_compact_stage_line(spec.name, agg))

        all_stage_reports["stages"][spec.name] = {
            "description": spec.description,
            "seeds": spec.seed_list,
            "config_overrides": spec.overrides,
            "aggregate": agg,
            "runs": run_reports,
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_stage_reports, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSaved staged summary: {summary_path}")


if __name__ == "__main__":
    main()
