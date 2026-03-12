"""Train Steane PPO once and evaluate the learned policy over a sweep of QEC cycles.

This script is intended for memory-decay style figures where:
  - a policy is trained once under one fixed noise condition
  - the trained policy is held fixed
  - evaluation-time `n_rounds` is swept over a chosen grid
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from ..ppo import train_ppo
from ..steane_adapter import SteaneAdapterConfig, SteaneOnlineSteeringSimulator, clipped_identity_action_mapper
from ..train import apply_google_paper_ppo_preset, evaluate_steane_policy_fn
from .eval_steane_ppo import (
    _build_ppo_config,
    _build_steane_components,
    _build_trace_finetune_args,
    build_arg_parser as build_benchmark_arg_parser,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for cycle-sweep evaluation."""
    parser = build_benchmark_arg_parser()
    parser.description = "Train Steane PPO once and evaluate fixed policies over a sweep of QEC cycles."
    parser.add_argument(
        "--cycle-sweep-rounds",
        type=str,
        default="5,10,15,20,25,30,35,40,45,50",
        help="Comma-separated evaluation-time n_rounds sweep.",
    )
    parser.add_argument(
        "--secondary-steane-measurement-bitflip-prob",
        type=float,
        default=-1.0,
        help=(
            "If >= 0, train a second policy with the same settings except for this "
            "measurement bit-flip probability, and include it in the sweep comparison."
        ),
    )
    parser.add_argument(
        "--primary-policy-label",
        type=str,
        default="trained_primary",
        help="Label used in saved reports for the primary trained policy.",
    )
    parser.add_argument(
        "--secondary-policy-label",
        type=str,
        default="trained_secondary",
        help="Label used in saved reports for the optional secondary trained policy.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for cycle-sweep evaluation."""
    return build_arg_parser().parse_args(argv)


def _clone_args(args: argparse.Namespace) -> argparse.Namespace:
    """Create a mutable shallow copy of argparse namespace."""
    return argparse.Namespace(**vars(args))


def _parse_round_sweep(spec: str) -> list[int]:
    """Parse comma-separated n_rounds sweep."""
    out: list[int] = []
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"cycle-sweep n_rounds must be > 0, got {value}")
        out.append(value)
    if not out:
        raise ValueError("cycle-sweep-rounds must contain at least one positive integer.")
    return out


def _logical_observable_proxy(success_rate: float) -> float:
    """Map commuting-basis success rate to a logical-observable proxy."""
    return float(2.0 * float(success_rate) - 1.0)


def _augment_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Attach derived logical proxy to evaluation summary."""
    out = dict(metrics)
    out["logical_observable_proxy"] = _logical_observable_proxy(float(metrics["success_rate"]))
    return out


def _rel_improvement(base: float, new: float) -> float:
    """Relative-improvement helper shared with benchmark reporting."""
    base_f = float(base)
    new_f = float(new)
    eps = 1e-12
    if base_f <= eps:
        return 0.0 if new_f <= eps else -1.0
    return float((base_f - new_f) / base_f)


def _evaluate_single_policy(
    model,
    steane_cfg: SteaneAdapterConfig,
    action_limit: float,
    episodes: int,
    eval_shots_per_step: int,
    seed: int,
) -> Dict[str, float]:
    """Evaluate one trained policy under one fixed Steane configuration."""
    eval_shots = int(eval_shots_per_step) if int(eval_shots_per_step) > 0 else int(steane_cfg.shots_per_step)
    eval_cfg = replace(
        steane_cfg,
        reset_drift_on_episode=True,
        shots_per_step=eval_shots,
        collect_traces=False,
        seed=int(seed),
    )
    simulator = SteaneOnlineSteeringSimulator(eval_cfg)
    action_mapper = lambda x: clipped_identity_action_mapper(x, action_limit=action_limit)
    device = next(model.parameters()).device
    use_cpu_fastpath = device.type == "cpu"

    def learned_policy(obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(-1)).unsqueeze(0)
        if not use_cpu_fastpath:
            obs_t = obs_t.to(device=device)
        with torch.inference_mode():
            theta = model.actor(obs_t).squeeze(0).cpu().numpy()
        return theta.astype(np.float32)

    return evaluate_steane_policy_fn(simulator, action_mapper, learned_policy, episodes=episodes)


def _evaluate_fixed_baselines(
    steane_cfg: SteaneAdapterConfig,
    action_limit: float,
    episodes: int,
    eval_shots_per_step: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Evaluate fixed-zero and random baselines once for one cycle count."""
    eval_shots = int(eval_shots_per_step) if int(eval_shots_per_step) > 0 else int(steane_cfg.shots_per_step)
    eval_cfg = replace(
        steane_cfg,
        reset_drift_on_episode=True,
        shots_per_step=eval_shots,
        collect_traces=False,
        seed=int(seed),
    )
    sim_fixed = SteaneOnlineSteeringSimulator(eval_cfg)
    sim_random = SteaneOnlineSteeringSimulator(eval_cfg)
    action_mapper = lambda x: clipped_identity_action_mapper(x, action_limit=action_limit)
    rng = np.random.default_rng(int(seed) + 2026)

    def fixed_zero_policy(_obs: np.ndarray) -> np.ndarray:
        return np.zeros(sim_fixed.action_dim, dtype=np.float32)

    def random_policy(_obs: np.ndarray) -> np.ndarray:
        return rng.uniform(-action_limit, action_limit, size=sim_random.action_dim).astype(np.float32)

    return {
        "fixed_zero": evaluate_steane_policy_fn(sim_fixed, action_mapper, fixed_zero_policy, episodes=episodes),
        "random_uniform": evaluate_steane_policy_fn(sim_random, action_mapper, random_policy, episodes=episodes),
    }


def _train_policy(args: argparse.Namespace) -> tuple[Any, Dict[str, list[float]], SteaneAdapterConfig, Any]:
    """Train one PPO policy under the provided args, including optional trace finetune."""
    steane_cfg, simulator, env = _build_steane_components(args)
    cfg = _build_ppo_config(args, simulator)
    model, history_phase1 = train_ppo(env, cfg)

    history_trace = None
    final_steane_cfg = steane_cfg
    final_cfg = cfg
    if int(args.trace_finetune_timesteps) > 0:
        ft_args = _build_trace_finetune_args(args)
        ft_steane_cfg, ft_simulator, ft_env = _build_steane_components(ft_args)
        final_cfg = _build_ppo_config(ft_args, ft_simulator)
        model, history_trace = train_ppo(ft_env, final_cfg, model=model)
        final_steane_cfg = ft_steane_cfg

    history = history_trace if history_trace is not None else history_phase1
    return model, history, final_steane_cfg, final_cfg


def _run_cycle_sweep(
    model,
    steane_cfg: SteaneAdapterConfig,
    args: argparse.Namespace,
    cycle_rounds: Sequence[int],
) -> list[Dict[str, Any]]:
    """Evaluate one trained policy and fixed baselines over n_rounds sweep."""
    rows: list[Dict[str, Any]] = []
    for rounds in cycle_rounds:
        eval_cfg = replace(steane_cfg, n_rounds=int(rounds))
        learned = _augment_metrics(
            _evaluate_single_policy(
                model=model,
                steane_cfg=eval_cfg,
                action_limit=float(args.action_limit),
                episodes=int(args.post_eval_episodes),
                eval_shots_per_step=int(args.eval_steane_shots_per_step),
                seed=int(args.seed),
            )
        )
        baselines = {
            name: _augment_metrics(metrics)
            for name, metrics in _evaluate_fixed_baselines(
                steane_cfg=eval_cfg,
                action_limit=float(args.action_limit),
                episodes=int(args.post_eval_episodes),
                eval_shots_per_step=int(args.eval_steane_shots_per_step),
                seed=int(args.seed),
            ).items()
        }
        timing = SteaneOnlineSteeringSimulator(eval_cfg).estimated_step_timing()
        fixed = baselines["fixed_zero"]
        rows.append(
            {
                "n_rounds": int(rounds),
                "n_steps": int(rounds) * 6,
                "nominal_circuit_timing_per_rl_step": {
                    "total_time_ns": float(timing.total_time_ns),
                    "active_time_ns": float(timing.active_time_ns),
                    "idle_time_ns": float(timing.idle_time_ns),
                    "n_operations": int(timing.n_operations),
                    "n_idle_windows": int(timing.n_idle_windows),
                    "assumes_single_prep_attempt": True,
                },
                "learned": learned,
                "fixed_zero": fixed,
                "random_uniform": baselines["random_uniform"],
                "improvement_vs_fixed_zero": {
                    "detector_rate": _rel_improvement(float(fixed["detector_rate"]), float(learned["detector_rate"])),
                    "ler_proxy": _rel_improvement(float(fixed["ler_proxy"]), float(learned["ler_proxy"])),
                },
            }
        )
    return rows


def run_cycle_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute cycle-sweep experiment and return structured report."""
    if args.google_paper_ppo_preset:
        apply_google_paper_ppo_preset(args)
    cycle_rounds = _parse_round_sweep(args.cycle_sweep_rounds)

    primary_model, primary_history, primary_steane_cfg, primary_ppo_cfg = _train_policy(args)
    primary_rows = _run_cycle_sweep(primary_model, primary_steane_cfg, args, cycle_rounds)

    secondary_report = None
    if float(args.secondary_steane_measurement_bitflip_prob) >= 0.0:
        secondary_args = _clone_args(args)
        secondary_args.steane_measurement_bitflip_prob = float(args.secondary_steane_measurement_bitflip_prob)
        secondary_model, secondary_history, secondary_steane_cfg, secondary_ppo_cfg = _train_policy(secondary_args)
        secondary_rows = _run_cycle_sweep(secondary_model, secondary_steane_cfg, secondary_args, cycle_rounds)
        secondary_report = {
            "label": str(args.secondary_policy_label),
            "train_args": vars(secondary_args),
            "steane_cfg": asdict(secondary_steane_cfg),
            "ppo_cfg": asdict(secondary_ppo_cfg),
            "final_mean_rollout_reward": float(secondary_history["mean_reward_rollout"][-1]),
            "cycle_sweep": secondary_rows,
        }

    report: Dict[str, Any] = {
        "cycle_rounds": [int(v) for v in cycle_rounds],
        "primary_policy": {
            "label": str(args.primary_policy_label),
            "train_args": vars(args),
            "steane_cfg": asdict(primary_steane_cfg),
            "ppo_cfg": asdict(primary_ppo_cfg),
            "final_mean_rollout_reward": float(primary_history["mean_reward_rollout"][-1]),
            "cycle_sweep": primary_rows,
        },
        "secondary_policy": secondary_report,
    }

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print cycle-sweep summary."""
    primary = report["primary_policy"]
    print("Cycle sweep finished.")
    print(f"Primary policy: {primary['label']}")
    print("n_rounds | logical_proxy | success_rate | ler_proxy | rel_improve_vs_fixed")
    for row in primary["cycle_sweep"]:
        print(
            f"{row['n_rounds']:>8d} | "
            f"{row['learned']['logical_observable_proxy']:+.6f} | "
            f"{row['learned']['success_rate']:.6f} | "
            f"{row['learned']['ler_proxy']:.6f} | "
            f"{row['improvement_vs_fixed_zero']['ler_proxy']:+.6f}"
        )
    if report["secondary_policy"] is not None:
        print(f"Secondary policy: {report['secondary_policy']['label']}")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    report = run_cycle_sweep(args)
    print_report(report)


if __name__ == "__main__":
    main()
