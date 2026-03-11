"""Train PPO on Steane adapter and report post-train policy performance.

This script is intentionally separate from `tests/` because it performs
stochastic, potentially long-running benchmarking rather than deterministic
unit checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Sequence, cast

import numpy as np
import torch

from ..config import PPOConfig
from ..codes.factory import available_code_families, build_code_components
from ..env import ExternalSimulatorEnv
from ..ppo import train_ppo
from ..steane_adapter import SteaneAdapterConfig, SteaneOnlineSteeringSimulator, clipped_identity_action_mapper
from ..train import (
    apply_google_paper_ppo_preset,
    evaluate_steane_policy_fn,
    make_steane_reward_fn,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI options for Steane PPO benchmark runs."""
    parser = argparse.ArgumentParser(description="Benchmark PPO performance on Steane RL environment.")

    # Preset and runtime controls.
    parser.add_argument("--code-family", choices=list(available_code_families()), default="steane")
    parser.add_argument("--google-paper-ppo-preset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--post-eval-episodes", type=int, default=30)
    parser.add_argument(
        "--eval-steane-shots-per-step",
        type=int,
        default=0,
        help="If >0, override shots_per_step only during post-train evaluation.",
    )
    parser.add_argument(
        "--trace-eval-episodes",
        type=int,
        default=0,
        help="If >0, run an additional trace-based evaluation pass.",
    )
    parser.add_argument(
        "--trace-eval-steane-shots-per-step",
        type=int,
        default=0,
        help="If >0, override shots_per_step only for trace-based evaluation.",
    )
    parser.add_argument("--save-json", type=str, default="")

    # PPO hyperparameters.
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=8_000)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=128)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--ppo-hidden-dim",
        type=int,
        default=128,
        help="Hidden width for actor/critic MLPs.",
    )
    parser.add_argument(
        "--ppo-use-layer-norm",
        action="store_true",
        help="Enable LayerNorm in hidden MLP blocks for actor/critic.",
    )
    parser.add_argument("--action-limit", type=float, default=2.0)
    parser.add_argument(
        "--trace-finetune-timesteps",
        type=int,
        default=0,
        help="If >0, run a second PPO phase using trace-enabled simulator feedback.",
    )
    parser.add_argument(
        "--trace-finetune-rollout-steps",
        type=int,
        default=0,
        help="If >0, override rollout steps in trace finetune phase.",
    )
    parser.add_argument(
        "--trace-finetune-shots-per-step",
        type=int,
        default=0,
        help="If >0, override shots_per_step in trace finetune phase.",
    )
    parser.add_argument(
        "--trace-finetune-n-rounds",
        type=int,
        default=0,
        help="If >0, override n_rounds in trace finetune phase.",
    )
    parser.add_argument(
        "--trace-finetune-learning-rate",
        type=float,
        default=0.0,
        help="If >0, override learning rate in trace finetune phase.",
    )
    parser.add_argument(
        "--trace-finetune-ent-coef",
        type=float,
        default=-1.0,
        help="If >=0, override entropy coefficient in trace finetune phase.",
    )

    # Steane simulator controls.
    parser.add_argument("--steane-n-rounds", type=int, default=25)
    parser.add_argument("--steane-shots-per-step", type=int, default=400)
    parser.add_argument("--steane-control-mode", choices=["global", "gate_specific"], default="gate_specific")
    parser.add_argument("--steane-control-dim", type=int, default=8)
    parser.add_argument("--steane-n-1q-control-slots", type=int, default=24)
    parser.add_argument("--steane-n-2q-control-slots", type=int, default=24)
    parser.add_argument("--steane-syndrome-mode", choices=["MV", "DE"], default="DE")
    parser.add_argument("--steane-stepping-mode", choices=["candidate_eval", "online_rounds"], default="candidate_eval")
    parser.add_argument("--steane-drift-period-steps", type=float, default=150.0)
    parser.add_argument("--steane-drift-amplitude", type=float, default=1.0)
    parser.add_argument("--steane-p1q-base", type=float, default=6.7e-4)
    parser.add_argument("--steane-p2q-base", type=float, default=2.7e-3)
    parser.add_argument("--steane-sensitivity-1q", type=float, default=2.0e-3)
    parser.add_argument("--steane-sensitivity-2q", type=float, default=5.0e-3)
    parser.add_argument("--steane-p-clip-max", type=float, default=0.3)
    parser.add_argument(
        "--steane-noise-channel",
        choices=[
            "auto",
            "google_global",
            "google_gate_specific",
            "idle_depolarizing",
            "parametric_google",
            "correlated_pauli_noise_channel",
            "composed_google_global_correlated",
            "composed_google_gate_specific_correlated",
        ],
        default="auto",
    )
    parser.add_argument("--steane-idle-p-total-per-idle", type=float, default=0.0)
    parser.add_argument("--steane-idle-px-weight", type=float, default=1.0)
    parser.add_argument("--steane-idle-py-weight", type=float, default=1.0)
    parser.add_argument("--steane-idle-pz-weight", type=float, default=1.0)
    parser.add_argument(
        "--steane-channel-corr-f",
        type=float,
        default=1.0e4,
        help=(
            "Correlation frequency f (Hz) for correlated/composed channels. "
            "Lower f means longer temporal correlation time."
        ),
    )
    parser.add_argument(
        "--steane-channel-corr-g",
        type=float,
        default=1.0,
        help=(
            "Overall channel-strength scale g for correlated/composed channels. "
            "Scales the nominal total Pauli error probability."
        ),
    )
    parser.add_argument(
        "--steane-channel-corr-g-mode",
        choices=["per_window", "per_circuit"],
        default="per_window",
        help=(
            "How to interpret correlated-channel g: "
            "per_window keeps legacy per-idle calibration; "
            "per_circuit normalizes by current circuit length."
        ),
    )
    # Deprecated aliases kept for backward compatibility with existing scripts.
    parser.add_argument("--steane-channel-regime-a", type=float, default=1.0)
    parser.add_argument("--steane-channel-regime-b", type=float, default=1.0)
    parser.add_argument("--steane-shot-workers", type=int, default=1)
    parser.add_argument("--steane-collect-traces", action="store_true")
    parser.add_argument("--steane-reset-drift-on-episode", action="store_true")
    parser.add_argument("--steane-expose-oracle-metrics", action="store_true")

    # Reward controls.
    parser.add_argument("--steane-reward-mode", choices=["legacy_success", "paper_surrogate"], default="paper_surrogate")
    parser.add_argument("--steane-action-penalty-coef", type=float, default=1e-3)
    parser.add_argument("--steane-miscal-penalty-coef", type=float, default=0.0)
    parser.add_argument("--steane-success-bonus-coef", type=float, default=0.0)

    return parser.parse_args(argv)


def _build_steane_components(
    args: argparse.Namespace,
) -> tuple[SteaneAdapterConfig, SteaneOnlineSteeringSimulator, ExternalSimulatorEnv]:
    """Build Steane simulator and PPO environment from args."""
    reward_fn = make_steane_reward_fn(
        mode=args.steane_reward_mode,
        action_penalty_coef=args.steane_action_penalty_coef,
        miscal_penalty_coef=args.steane_miscal_penalty_coef,
        success_bonus_coef=args.steane_success_bonus_coef,
    )
    components = build_code_components(args, reward_fn=reward_fn)
    if components.code_family != "steane":
        raise ValueError("eval_steane_ppo currently supports only code-family 'steane'.")
    steane_cfg = cast(SteaneAdapterConfig, components.code_cfg)
    simulator = cast(SteaneOnlineSteeringSimulator, components.simulator)
    env = cast(ExternalSimulatorEnv, components.env)
    return steane_cfg, simulator, env


def _build_ppo_config(args: argparse.Namespace, simulator: SteaneOnlineSteeringSimulator) -> PPOConfig:
    """Assemble PPOConfig from CLI args and simulator dimensions."""
    return PPOConfig(
        obs_dim=simulator.obs_dim,
        theta_dim=simulator.action_dim,
        max_steps=args.max_steps,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.ppo_update_epochs,
        minibatch_size=args.ppo_minibatch_size,
        learning_rate=args.ppo_learning_rate,
        ent_coef=args.ppo_ent_coef,
        hidden_dim=args.ppo_hidden_dim,
        use_layer_norm=bool(args.ppo_use_layer_norm),
        device=args.device,
        seed=args.seed,
    )


def _evaluate_policies(
    model,
    cfg: PPOConfig,
    steane_cfg: SteaneAdapterConfig,
    action_limit: float,
    episodes: int,
    eval_shots_per_step: int = 0,
    eval_collect_traces: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run fair learned/fixed/random comparison with drift-reset evaluation sims."""
    eval_shots = int(eval_shots_per_step) if int(eval_shots_per_step) > 0 else int(steane_cfg.shots_per_step)
    eval_cfg = replace(
        steane_cfg,
        reset_drift_on_episode=True,
        shots_per_step=eval_shots,
        collect_traces=bool(eval_collect_traces),
    )
    sim_learned = SteaneOnlineSteeringSimulator(eval_cfg)
    sim_fixed = SteaneOnlineSteeringSimulator(eval_cfg)
    sim_random = SteaneOnlineSteeringSimulator(eval_cfg)
    action_mapper = lambda x: clipped_identity_action_mapper(x, action_limit=action_limit)
    rng = np.random.default_rng(int(cfg.seed) + 2026)
    device = torch.device(cfg.device)
    use_cpu_fastpath = device.type == "cpu"

    def learned_policy(obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(-1)).unsqueeze(0)
        if not use_cpu_fastpath:
            obs_t = obs_t.to(device=device)
        with torch.inference_mode():
            theta = model.actor(obs_t).squeeze(0).cpu().numpy()
        return theta.astype(np.float32)

    def fixed_zero_policy(_obs: np.ndarray) -> np.ndarray:
        return np.zeros(sim_learned.action_dim, dtype=np.float32)

    def random_policy(_obs: np.ndarray) -> np.ndarray:
        # Random baseline uses same bounded action range as runtime mapper.
        return rng.uniform(-action_limit, action_limit, size=sim_learned.action_dim).astype(np.float32)

    learned = evaluate_steane_policy_fn(sim_learned, action_mapper, learned_policy, episodes=episodes)
    fixed = evaluate_steane_policy_fn(sim_fixed, action_mapper, fixed_zero_policy, episodes=episodes)
    random = evaluate_steane_policy_fn(sim_random, action_mapper, random_policy, episodes=episodes)
    return {"learned": learned, "fixed_zero": fixed, "random_uniform": random}


def _rel_improvement(base: float, new: float) -> float:
    base_f = float(base)
    new_f = float(new)
    eps = 1e-12
    # Relative improvement is ill-defined when baseline is numerically zero.
    # Use a bounded convention:
    #   - 0.0 when both are ~0
    #   - -1.0 when baseline is ~0 but new is worse (>0)
    if base_f <= eps:
        return 0.0 if new_f <= eps else -1.0
    return float((base_f - new_f) / base_f)


def _clone_args(args: argparse.Namespace) -> argparse.Namespace:
    """Create a mutable copy of argparse namespace."""
    return argparse.Namespace(**vars(args))


def _build_trace_finetune_args(base_args: argparse.Namespace) -> argparse.Namespace:
    """Build phase-2 trace-finetune args from phase-1 args."""
    ft = _clone_args(base_args)
    ft.steane_collect_traces = True
    if int(base_args.trace_finetune_rollout_steps) > 0:
        ft.rollout_steps = int(base_args.trace_finetune_rollout_steps)
    if int(base_args.trace_finetune_shots_per_step) > 0:
        ft.steane_shots_per_step = int(base_args.trace_finetune_shots_per_step)
    if int(base_args.trace_finetune_n_rounds) > 0:
        ft.steane_n_rounds = int(base_args.trace_finetune_n_rounds)
    if float(base_args.trace_finetune_learning_rate) > 0.0:
        ft.ppo_learning_rate = float(base_args.trace_finetune_learning_rate)
    if float(base_args.trace_finetune_ent_coef) >= 0.0:
        ft.ppo_ent_coef = float(base_args.trace_finetune_ent_coef)
    ft.total_timesteps = int(base_args.trace_finetune_timesteps)
    return ft


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute one Steane PPO benchmark run and return structured report."""
    if args.google_paper_ppo_preset:
        apply_google_paper_ppo_preset(args)

    # Phase-1: fast proxy training.
    steane_cfg, simulator, env = _build_steane_components(args)
    cfg = _build_ppo_config(args, simulator)
    model, history_phase1 = train_ppo(env, cfg)

    history_trace = None
    trace_ft_cfg = None
    # Phase-2 (optional): trace-based finetuning for richer detector signal.
    if int(args.trace_finetune_timesteps) > 0:
        ft_args = _build_trace_finetune_args(args)
        ft_steane_cfg, ft_simulator, ft_env = _build_steane_components(ft_args)
        trace_ft_cfg = _build_ppo_config(ft_args, ft_simulator)
        model, history_trace = train_ppo(ft_env, trace_ft_cfg, model=model)
        # Keep final report's steane cfg aligned to phase-2 if finetune was used.
        steane_cfg = ft_steane_cfg
        cfg = trace_ft_cfg

    step_timing = simulator.estimated_step_timing() if history_trace is None else ft_simulator.estimated_step_timing()

    eval_metrics = _evaluate_policies(
        model=model,
        cfg=cfg,
        steane_cfg=steane_cfg,
        action_limit=args.action_limit,
        episodes=args.post_eval_episodes,
        eval_shots_per_step=int(args.eval_steane_shots_per_step),
    )

    trace_eval_metrics = None
    if int(args.trace_eval_episodes) > 0:
        trace_eval_metrics = _evaluate_policies(
            model=model,
            cfg=cfg,
            steane_cfg=steane_cfg,
            action_limit=args.action_limit,
            episodes=int(args.trace_eval_episodes),
            eval_shots_per_step=int(args.trace_eval_steane_shots_per_step),
            eval_collect_traces=True,
        )

    learned = eval_metrics["learned"]
    fixed = eval_metrics["fixed_zero"]
    random = eval_metrics["random_uniform"]
    final_rollout_reward = (
        float(history_trace["mean_reward_rollout"][-1])
        if history_trace is not None
        else float(history_phase1["mean_reward_rollout"][-1])
    )
    report: Dict[str, Any] = {
        "args": vars(args),
        "steane_cfg": asdict(steane_cfg),
        "ppo_cfg": asdict(cfg),
        "nominal_circuit_timing_per_rl_step": {
            "total_time_ns": float(step_timing.total_time_ns),
            "active_time_ns": float(step_timing.active_time_ns),
            "idle_time_ns": float(step_timing.idle_time_ns),
            "n_operations": int(step_timing.n_operations),
            "n_idle_windows": int(step_timing.n_idle_windows),
            "assumes_single_prep_attempt": True,
        },
        "phase1_mean_rollout_reward": float(history_phase1["mean_reward_rollout"][-1]),
        "trace_finetune_enabled": bool(history_trace is not None),
        "trace_finetune_ppo_cfg": asdict(trace_ft_cfg) if trace_ft_cfg is not None else None,
        "trace_finetune_mean_rollout_reward": (
            float(history_trace["mean_reward_rollout"][-1]) if history_trace is not None else None
        ),
        "final_mean_rollout_reward": final_rollout_reward,
        "eval_metrics": eval_metrics,
        "trace_eval_metrics": trace_eval_metrics,
        "improvement_vs_fixed_zero": {
            "detector_rate": _rel_improvement(fixed["detector_rate"], learned["detector_rate"]),
            "ler_proxy": _rel_improvement(fixed["ler_proxy"], learned["ler_proxy"]),
        },
        "improvement_vs_random_uniform": {
            "detector_rate": _rel_improvement(random["detector_rate"], learned["detector_rate"]),
            "ler_proxy": _rel_improvement(random["ler_proxy"], learned["ler_proxy"]),
        },
    }

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print key benchmark outputs for interactive runs."""
    learned = report["eval_metrics"]["learned"]
    fixed = report["eval_metrics"]["fixed_zero"]
    random = report["eval_metrics"]["random_uniform"]
    print("Benchmark finished.")
    print(f"Final mean rollout reward: {report['final_mean_rollout_reward']:.6f}")
    print(
        "Learned metrics: "
        f"DR={learned['detector_rate']:.6f}, success={learned['success_rate']:.6f}, LER~={learned['ler_proxy']:.6f}"
    )
    print(
        "Fixed-zero metrics: "
        f"DR={fixed['detector_rate']:.6f}, success={fixed['success_rate']:.6f}, LER~={fixed['ler_proxy']:.6f}"
    )
    print(
        "Random metrics: "
        f"DR={random['detector_rate']:.6f}, success={random['success_rate']:.6f}, LER~={random['ler_proxy']:.6f}"
    )
    print(
        "Improvement vs fixed-zero: "
        f"DR={100.0 * report['improvement_vs_fixed_zero']['detector_rate']:+.2f}%, "
        f"LER~={100.0 * report['improvement_vs_fixed_zero']['ler_proxy']:+.2f}%"
    )


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    print_report(report)
    if args.save_json:
        print(f"Saved benchmark report: {args.save_json}")


if __name__ == "__main__":
    main()
