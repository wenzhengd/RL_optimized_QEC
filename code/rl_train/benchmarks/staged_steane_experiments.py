"""Multi-stage Steane PPO experiment driver.

Stages follow a practical progression:
  1) sanity: tiny compute budget, verify pipeline runs end-to-end
  2) pilot: moderate budget, check whether RL shows positive signal
  3) scale: larger budget, probe improvement under heavier training/eval
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from quantum_simulation.noise_channels import available_steane_noise_channels

from ..codes.factory import available_code_families
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
        help=(
            "Comma-separated stage IDs to run. Use 'all' to run all stages "
            "from the selected stage spec source."
        ),
    )
    parser.add_argument(
        "--stage-specs-json",
        type=str,
        default="",
        help=(
            "Optional JSON file path providing custom stage specs. "
            "Schema: {'stages': {'id': {'name','description','seed_list','overrides'}}}."
        ),
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
        "--seed-workers",
        type=int,
        default=1,
        help="Parallel worker processes over seeds inside each stage. 1 means sequential.",
    )
    parser.add_argument(
        "--print-each-run",
        action="store_true",
        help="Print full per-run report blocks in addition to compact summary.",
    )
    parser.add_argument(
        "--code-family",
        choices=list(available_code_families()),
        default=None,
        help="Optional global code family override applied to every selected stage.",
    )
    parser.add_argument(
        "--steane-noise-channel",
        choices=list(available_steane_noise_channels()),
        default=None,
        help="Optional global Steane noise channel override applied to every selected stage.",
    )
    parser.add_argument(
        "--base-overrides-json",
        type=str,
        default="",
        help=(
            "Optional JSON file path containing eval_steane_ppo arg overrides "
            "applied to every selected stage."
        ),
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
        "5": StageSpec(
            name="stage5_power_stats",
            description=(
                "Statistics-focused run: keep training setup fixed, increase seeds "
                "and evaluation sampling to improve confidence in effect estimates."
            ),
            seed_list=[60 + seed_offset + i for i in range(20)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Keep training workload identical to stage4 for fair ablation.
                "total_timesteps": 512,
                "rollout_steps": 32,
                "steane_n_rounds": 4,
                "steane_shots_per_step": 4,
                # Increase only statistics/evaluation budget.
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "6": StageSpec(
            name="stage6_trace_finetune",
            description=(
                "Signal-quality run: fast phase-1 training plus short trace-based "
                "phase-2 finetune, with stronger evaluation statistics."
            ),
            seed_list=[90 + seed_offset + i for i in range(12)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Phase-1 fast training (same family as stage4/5 for fair comparison).
                "total_timesteps": 512,
                "rollout_steps": 32,
                "steane_n_rounds": 4,
                "steane_shots_per_step": 4,
                # Phase-2 trace-based finetune (small budget).
                "trace_finetune_timesteps": 64,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 8,
                "trace_finetune_n_rounds": 4,
                # Evaluation budget.
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                # Optional trace-only eval for higher-fidelity detector metric.
                "trace_eval_episodes": 8,
                "trace_eval_steane_shots_per_step": 32,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "7": StageSpec(
            name="stage7_progressive_scale",
            description=(
                "Progressive scale-up: keep trace-finetune workflow and increase "
                "training/data budget moderately for stronger policy learning."
            ),
            seed_list=[120 + seed_offset + i for i in range(10)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Phase-1 training: moderate scale-up vs stage6.
                "total_timesteps": 1024,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 6,
                # Phase-2 trace finetune: still small, but stronger than stage6.
                "trace_finetune_timesteps": 192,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 12,
                "trace_finetune_n_rounds": 6,
                # Keep high-statistics eval.
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                "trace_eval_episodes": 8,
                "trace_eval_steane_shots_per_step": 32,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "8": StageSpec(
            name="stage8_scale_x3",
            description=(
                "Scale curve point A: ~3x per-seed compute vs stage7 while "
                "preserving PPO and trace-finetune workflow."
            ),
            seed_list=[140 + seed_offset + i for i in range(10)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Phase-1: larger timesteps/shots, same rollout granularity.
                "total_timesteps": 3072,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 8,
                # Phase-2 trace finetune: scaled up but still cheaper than phase-1.
                "trace_finetune_timesteps": 384,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 16,
                "trace_finetune_n_rounds": 6,
                # Keep high-stat eval for comparable confidence.
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                "trace_eval_episodes": 8,
                "trace_eval_steane_shots_per_step": 32,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "9": StageSpec(
            name="stage9_scale_x10",
            description=(
                "Scale curve point B: around an order-of-magnitude per-seed "
                "compute vs stage7 with stronger train/eval sampling."
            ),
            seed_list=[160 + seed_offset + i for i in range(8)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Phase-1: major data scale increase.
                "total_timesteps": 8192,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 10,
                # Phase-2 trace finetune: increase both horizon and shot count.
                "trace_finetune_timesteps": 1024,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 20,
                "trace_finetune_n_rounds": 6,
                # Increase eval sampling to tighten uncertainty at this scale.
                "post_eval_episodes": 48,
                "eval_steane_shots_per_step": 96,
                "trace_eval_episodes": 12,
                "trace_eval_steane_shots_per_step": 48,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "10": StageSpec(
            name="stage10_scale_x30",
            description=(
                "Scale curve point C: high-compute run targeting ~30x training "
                "data regime vs stage7 while keeping algorithm unchanged."
            ),
            seed_list=[180 + seed_offset + i for i in range(6)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Phase-1: dominant compute increase for scaling-law signal.
                "total_timesteps": 24576,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 12,
                # Phase-2 trace finetune: proportional increase, still secondary.
                "trace_finetune_timesteps": 2048,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 24,
                "trace_finetune_n_rounds": 6,
                # Keep stronger evaluation statistics.
                "post_eval_episodes": 64,
                "eval_steane_shots_per_step": 128,
                "trace_eval_episodes": 16,
                "trace_eval_steane_shots_per_step": 64,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "11": StageSpec(
            name="stage11_arch_mlp",
            description=(
                "Architecture fairness run: keep stage8 data budget/protocol and "
                "switch to a wider LayerNorm MLP policy/value network."
            ),
            seed_list=[220 + seed_offset + i for i in range(10)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Keep stage8 training budget for apples-to-apples comparison.
                "total_timesteps": 3072,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 8,
                "trace_finetune_timesteps": 384,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 16,
                "trace_finetune_n_rounds": 6,
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                "trace_eval_episodes": 8,
                "trace_eval_steane_shots_per_step": 32,
                # Architecture-only change.
                "ppo_hidden_dim": 256,
                "ppo_use_layer_norm": True,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "12": StageSpec(
            name="stage12_arch_mlp_tune_a",
            description=(
                "Hyperparameter tune A (same stage8 budget): wider LayerNorm MLP "
                "with lower PPO learning rates for stability."
            ),
            seed_list=[240 + seed_offset + i for i in range(5)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Same budget/protocol as stage8 and stage11.
                "total_timesteps": 3072,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 8,
                "trace_finetune_timesteps": 384,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 16,
                "trace_finetune_n_rounds": 6,
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                "trace_eval_episodes": 8,
                "trace_eval_steane_shots_per_step": 32,
                # Architecture fixed; tune optimizer only.
                "ppo_hidden_dim": 256,
                "ppo_use_layer_norm": True,
                "ppo_learning_rate": 1e-4,
                "ppo_ent_coef": 0.01,
                "trace_finetune_learning_rate": 5e-5,
                "trace_finetune_ent_coef": 0.01,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "13": StageSpec(
            name="stage13_arch_mlp_tune_b",
            description=(
                "Hyperparameter tune B (same stage8 budget): lower learning rate "
                "with reduced entropy pressure in phase-1/phase-2."
            ),
            seed_list=[250 + seed_offset + i for i in range(5)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Same budget/protocol as stage8 and stage11.
                "total_timesteps": 3072,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 8,
                "trace_finetune_timesteps": 384,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 16,
                "trace_finetune_n_rounds": 6,
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                "trace_eval_episodes": 8,
                "trace_eval_steane_shots_per_step": 32,
                # Architecture fixed; tune optimizer only.
                "ppo_hidden_dim": 256,
                "ppo_use_layer_norm": True,
                "ppo_learning_rate": 1e-4,
                "ppo_ent_coef": 0.005,
                "trace_finetune_learning_rate": 5e-5,
                "trace_finetune_ent_coef": 0.005,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "14": StageSpec(
            name="stage14_arch_mlp_tune_c",
            description=(
                "Hyperparameter tune C (same stage8 budget): moderate learning rate "
                "plus more PPO update passes for the wider LayerNorm MLP."
            ),
            seed_list=[260 + seed_offset + i for i in range(5)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Same budget/protocol as stage8 and stage11.
                "total_timesteps": 3072,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 8,
                "trace_finetune_timesteps": 384,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 16,
                "trace_finetune_n_rounds": 6,
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                "trace_eval_episodes": 8,
                "trace_eval_steane_shots_per_step": 32,
                # Architecture fixed; tune optimizer only.
                "ppo_hidden_dim": 256,
                "ppo_use_layer_norm": True,
                "ppo_learning_rate": 1.5e-4,
                "ppo_ent_coef": 0.005,
                "ppo_update_epochs": 6,
                "ppo_minibatch_size": 64,
                "trace_finetune_learning_rate": 7.5e-5,
                "trace_finetune_ent_coef": 0.005,
                "steane_shot_workers": 1,
                "steane_collect_traces": False,
            },
        ),
        "15": StageSpec(
            name="stage15_arch_mlp_tuned_confirm",
            description=(
                "Confirmation run after tuning: keep stage8 budget and use selected "
                "wider LayerNorm MLP hyperparameters with 10 seeds."
            ),
            seed_list=[270 + seed_offset + i for i in range(10)],
            overrides={
                "google_paper_ppo_preset": False,
                "device": device,
                # Same budget/protocol as stage8 and stage11.
                "total_timesteps": 3072,
                "rollout_steps": 32,
                "steane_n_rounds": 6,
                "steane_shots_per_step": 8,
                "trace_finetune_timesteps": 384,
                "trace_finetune_rollout_steps": 16,
                "trace_finetune_shots_per_step": 16,
                "trace_finetune_n_rounds": 6,
                "post_eval_episodes": 32,
                "eval_steane_shots_per_step": 64,
                "trace_eval_episodes": 8,
                "trace_eval_steane_shots_per_step": 32,
                # Selected initial tuned setting (updated after stage12-14 comparison if needed).
                "ppo_hidden_dim": 256,
                "ppo_use_layer_norm": True,
                "ppo_learning_rate": 1e-4,
                "ppo_ent_coef": 0.01,
                "trace_finetune_learning_rate": 5e-5,
                "trace_finetune_ent_coef": 0.01,
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


def _build_base_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Build global overrides shared by all selected stages."""
    out: Dict[str, Any] = {}
    if args.base_overrides_json:
        p = Path(args.base_overrides_json)
        if not p.exists():
            raise FileNotFoundError(f"--base-overrides-json not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--base-overrides-json must contain a JSON object.")
        out.update(payload)
    if args.code_family is not None:
        out["code_family"] = str(args.code_family)
    if args.steane_noise_channel is not None:
        out["steane_noise_channel"] = str(args.steane_noise_channel)
    return out


def _parse_stage_ids(stages_arg: str, available_ids: Sequence[str]) -> List[str]:
    """Parse requested stage IDs from CLI argument."""
    text = str(stages_arg).strip()
    if text.lower() == "all":
        return list(available_ids)
    return [s.strip() for s in text.split(",") if s.strip()]


def _load_stage_specs_from_json(path: str, seed_offset: int, device: str) -> Dict[str, StageSpec]:
    """Load stage specs from a JSON config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--stage-specs-json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--stage-specs-json must contain a JSON object.")
    raw_stages = payload.get("stages", payload)
    if not isinstance(raw_stages, dict) or not raw_stages:
        raise ValueError("--stage-specs-json must provide a non-empty stages mapping.")

    out: Dict[str, StageSpec] = {}
    for sid, stage_payload in raw_stages.items():
        if not isinstance(stage_payload, dict):
            raise ValueError(f"Stage '{sid}' must be a JSON object.")
        name = str(stage_payload.get("name", sid))
        description = str(stage_payload.get("description", ""))
        raw_seed_list = stage_payload.get("seed_list")
        if not isinstance(raw_seed_list, list) or not raw_seed_list:
            raise ValueError(f"Stage '{sid}' must define a non-empty 'seed_list' array.")
        seed_list = [int(x) + int(seed_offset) for x in raw_seed_list]
        overrides = stage_payload.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Stage '{sid}' field 'overrides' must be a JSON object.")
        merged_overrides = dict(overrides)
        merged_overrides.setdefault("device", str(device))
        out[str(sid)] = StageSpec(
            name=name,
            description=description,
            seed_list=seed_list,
            overrides=merged_overrides,
        )
    return out


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


def _run_single_seed(
    seed: int,
    overrides: Dict[str, Any],
    save_json: str,
) -> Dict[str, Any]:
    """Worker entry for one seed benchmark run.

    Uses process-level parallelism to bypass Python GIL limits in simulator-heavy loops.
    """
    # Avoid thread oversubscription inside each process.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        # Keep benchmark runnable even if torch thread controls are unavailable.
        pass

    run_args = parse_eval_args([])
    run_args = _apply_overrides(run_args, overrides)
    run_args.seed = int(seed)
    run_args.save_json = str(save_json)
    return run_benchmark(run_args)


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
    if args.stage_specs_json:
        stage_specs = _load_stage_specs_from_json(
            path=args.stage_specs_json,
            seed_offset=args.seed_offset,
            device=args.device,
        )
        stage_spec_source = str(args.stage_specs_json)
    else:
        stage_specs = _default_stage_specs(seed_offset=args.seed_offset, device=args.device)
        stage_spec_source = "default"
    stage_order = _parse_stage_ids(args.stages, available_ids=list(stage_specs.keys()))
    base_overrides = _build_base_overrides(args)
    unknown = [s for s in stage_order if s not in stage_specs]
    if unknown:
        raise ValueError(
            f"Unknown stage ids: {unknown}. Allowed from current stage spec source: {list(stage_specs.keys())}."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stage_reports: Dict[str, Any] = {
        "stage_spec_source": stage_spec_source,
        "selected_stage_ids": stage_order,
        "base_overrides": base_overrides,
        "stages": {},
    }
    if args.seed_workers < 1:
        raise ValueError("--seed-workers must be >= 1.")
    if base_overrides:
        print(f"Applying global overrides to all selected stages: {base_overrides}")

    for sid in stage_order:
        spec = stage_specs[sid]
        print(f"\n=== {spec.name} ===")
        print(spec.description)

        run_reports: List[Dict[str, Any]] = []
        merged_overrides = dict(spec.overrides)
        merged_overrides.update(base_overrides)
        stage_dir = out_dir / spec.name
        stage_dir.mkdir(parents=True, exist_ok=True)

        workers = min(int(args.seed_workers), len(spec.seed_list))
        if workers <= 1:
            for seed in spec.seed_list:
                report = _run_single_seed(
                    seed=int(seed),
                    overrides=merged_overrides,
                    save_json=str(stage_dir / f"seed_{seed}.json"),
                )
                run_reports.append(report)
                print(
                    f"seed={seed}: "
                    f"learned_success={100.0 * report['eval_metrics']['learned']['success_rate']:.2f}%, "
                    f"improve(LER~)={100.0 * report['improvement_vs_fixed_zero']['ler_proxy']:+.2f}%"
                )
                if args.print_each_run:
                    print_report(report)
        else:
            print(f"Running seeds in parallel with {workers} workers.")
            reports_by_seed: Dict[int, Dict[str, Any]] = {}
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(
                        _run_single_seed,
                        int(seed),
                        merged_overrides,
                        str(stage_dir / f"seed_{seed}.json"),
                    ): int(seed)
                    for seed in spec.seed_list
                }
                for fut in as_completed(futures):
                    seed = futures[fut]
                    report = fut.result()
                    reports_by_seed[seed] = report
                    print(
                        f"seed={seed}: "
                        f"learned_success={100.0 * report['eval_metrics']['learned']['success_rate']:.2f}%, "
                        f"improve(LER~)={100.0 * report['improvement_vs_fixed_zero']['ler_proxy']:+.2f}%"
                    )

            # Keep downstream aggregation/report order deterministic.
            run_reports = [reports_by_seed[int(seed)] for seed in spec.seed_list]
            if args.print_each_run:
                for report in run_reports:
                    print_report(report)

        agg = _aggregate_stage_metrics(run_reports)
        print(_compact_stage_line(spec.name, agg))

        all_stage_reports["stages"][spec.name] = {
            "description": spec.description,
            "seeds": spec.seed_list,
            "config_overrides": merged_overrides,
            "aggregate": agg,
            "runs": run_reports,
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_stage_reports, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSaved staged summary: {summary_path}")


if __name__ == "__main__":
    main()
