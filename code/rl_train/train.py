"""
Entry script for your custom simulator + PPO setup.

Replace TODO sections with your task definition.
"""

import argparse
from dataclasses import replace
from typing import Callable, Dict, cast

import numpy as np
import torch

from .config import PPOConfig
from .codes.factory import available_code_families, build_code_components
from .env import ExternalSimulatorEnv, identity_action_mapper
from .example_simulator import ExampleLinearSimulator
from .interfaces import SimulatorTransition
from .ppo import train_ppo
from .steane_adapter import (
    SteaneAdapterConfig,
    SteaneOnlineSteeringSimulator,
)


def example_reward_fn(
    obs_t: np.ndarray,
    theta_t: np.ndarray,
    action_t: np.ndarray,
    obs_tp1: np.ndarray,
    simulator_feedback: Dict,
    timestep: int,
) -> float:
    """
    Runnable example reward.
    Replace this with your own reward formula when ready.

    This example reward is intentionally simple:
      - penalize distance to a fixed target state
      - add small control-energy penalty on action magnitude
    """
    _ = (obs_t, theta_t, action_t, timestep)
    return -float(simulator_feedback["distance_to_target"]) - 1e-3 * float(np.square(action_t).sum())


class YourSimulator:
    """
    TODO: Replace with your real callable simulator.

    Required interface:
      reset() -> np.ndarray
      step(action: np.ndarray) -> SimulatorTransition(next_obs, info, done)

    Keep this class small. Put simulator-heavy logic in a dedicated module
    if your project grows.
    """

    def reset(self) -> np.ndarray:
        """Reset simulator state and return initial observation."""
        raise NotImplementedError("TODO: hook your simulator here.")

    def step(self, action: np.ndarray) -> SimulatorTransition:
        """Apply action and return one simulator transition."""
        _ = action
        raise NotImplementedError("TODO: hook your simulator here.")


def steane_reward_fn(
    obs_t: np.ndarray,
    theta_t: np.ndarray,
    action_t: np.ndarray,
    obs_tp1: np.ndarray,
    simulator_feedback: Dict,
    timestep: int,
) -> float:
    """Reward for Steane online steering.

    Objective:
      - maximize success rate
      - mildly penalize large actions

    Note:
      This legacy reward intentionally does not rely on oracle-only metrics.
    """
    _ = (theta_t, timestep)
    success_abs = float(obs_tp1[0])
    success_delta = float(obs_tp1[0] - obs_t[0])
    action_penalty = 1e-3 * float(np.square(action_t).sum())
    return success_abs + 0.5 * success_delta - action_penalty


def make_steane_reward_fn(
    mode: str,
    action_penalty_coef: float,
    miscal_penalty_coef: float,
    success_bonus_coef: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, int], float]:
    """Build a Steane reward callback.

    Modes:
      - legacy_success: original success-centric shaping.
      - paper_surrogate: detector-rate surrogate objective inspired by Google RL QEC paper.
    """

    if mode == "legacy_success":
        return steane_reward_fn

    if mode != "paper_surrogate":
        raise ValueError(f"Unknown reward mode: {mode}")

    def _reward(
        obs_t: np.ndarray,
        theta_t: np.ndarray,
        action_t: np.ndarray,
        obs_tp1: np.ndarray,
        simulator_feedback: Dict,
        timestep: int,
    ) -> float:
        _ = (obs_t, theta_t, timestep)
        detector_rates = simulator_feedback.get("detector_rates", simulator_feedback.get("stabilizer_means", []))
        detector_rates_arr = np.asarray(detector_rates, dtype=float).reshape(-1)
        if detector_rates_arr.size > 0:
            surrogate_cost = float(np.mean(detector_rates_arr))
        else:
            surrogate_cost = 1.0 - float(obs_tp1[0])

        action_penalty = float(action_penalty_coef) * float(np.square(action_t).sum())
        miscal_penalty = float(miscal_penalty_coef) * float(simulator_feedback.get("miscalibration_mse", 0.0))
        success_bonus = float(success_bonus_coef) * float(simulator_feedback.get("success_rate", obs_tp1[0]))
        return -surrogate_cost + success_bonus - action_penalty - miscal_penalty

    return _reward


def apply_google_paper_ppo_preset(args: argparse.Namespace) -> None:
    """Apply paper-inspired PPO defaults while keeping PPO as optimizer."""
    args.steane_control_mode = "gate_specific"
    args.steane_noise_channel = "google_gate_specific"
    args.steane_n_1q_control_slots = 32
    args.steane_n_2q_control_slots = 32
    args.steane_stepping_mode = "candidate_eval"
    args.steane_n_rounds = 25
    args.steane_syndrome_mode = "DE"
    args.steane_reward_mode = "paper_surrogate"

    # Keep runtime manageable in simulation while preserving paper-style structure.
    args.steane_shots_per_step = 400
    args.max_steps = 1
    args.rollout_steps = 40
    args.total_timesteps = 8_000
    args.ppo_ent_coef = 0.01
    # Disable oracle-dependent penalty terms by default.
    args.steane_miscal_penalty_coef = 0.0
    args.steane_expose_oracle_metrics = False
    # Default to fast summary mode for iterative RL experiments.
    args.steane_collect_traces = False
    args.steane_shot_workers = 1


def _summarize_detector_rate(info: Dict, fallback_success_rate: float) -> float:
    """Return detector-rate proxy from simulator info.

    If explicit detector vector is missing, fall back to `1-success_rate`.
    """
    detector_rates = np.asarray(info.get("detector_rates", []), dtype=float).reshape(-1)
    if detector_rates.size > 0:
        return float(np.mean(detector_rates))
    return 1.0 - float(fallback_success_rate)


def evaluate_steane_policy_fn(
    simulator: SteaneOnlineSteeringSimulator,
    action_mapper: Callable[[np.ndarray], np.ndarray],
    policy_fn: Callable[[np.ndarray], np.ndarray],
    episodes: int,
) -> Dict[str, float]:
    """Evaluate one policy family on Steane simulator and return DR/LER summary.

    Protocol:
      - each episode samples a candidate policy from `policy_fn(obs0)`
      - that candidate is held fixed for the full candidate evaluation run
      - metrics are averaged across episodes
    """
    if episodes <= 0:
        raise ValueError("episodes must be > 0.")

    det_rates = []
    success_rates = []
    for _ in range(int(episodes)):
        obs0 = simulator.reset()
        theta = np.asarray(policy_fn(obs0), dtype=np.float32).reshape(-1)
        action = np.asarray(action_mapper(theta), dtype=np.float32).reshape(-1)

        if simulator.cfg.stepping_mode == "candidate_eval":
            max_sim_steps = 1
        else:
            max_sim_steps = max(1, int(simulator.cfg.n_rounds))

        episode_det = []
        episode_success = []
        for _step in range(max_sim_steps):
            transition = simulator.step(action)
            success = float(transition.info.get("success_rate", transition.next_obs[0]))
            episode_success.append(success)
            episode_det.append(_summarize_detector_rate(transition.info, fallback_success_rate=success))
            if transition.done:
                break

        det_rates.append(float(np.mean(episode_det)) if episode_det else 1.0)
        success_rates.append(float(np.mean(episode_success)) if episode_success else 0.0)

    dr = float(np.mean(det_rates))
    success = float(np.mean(success_rates))
    ler = 1.0 - success
    return {"detector_rate": dr, "success_rate": success, "ler_proxy": ler}


def main() -> None:
    """
    Build config, instantiate environment, and run PPO training.

    The script contains two mutually exclusive setup paths:
      - Option A: real simulator integration (recommended for your task)
      - Option B: toy simulator sanity check (runs once torch is installed)
    """
    args = parse_args()
    if args.google_paper_ppo_preset:
        apply_google_paper_ppo_preset(args)

    simulator = None
    action_mapper = None
    steane_cfg = None
    if args.backend == "steane":
        steane_reward = make_steane_reward_fn(
            mode=args.steane_reward_mode,
            action_penalty_coef=args.steane_action_penalty_coef,
            miscal_penalty_coef=args.steane_miscal_penalty_coef,
            success_bonus_coef=args.steane_success_bonus_coef,
        )
        components = build_code_components(args, reward_fn=steane_reward)
        if components.code_family != "steane":
            raise ValueError("train.py post-eval currently supports only steane code family.")
        steane_cfg = cast(SteaneAdapterConfig, components.code_cfg)
        simulator = cast(SteaneOnlineSteeringSimulator, components.simulator)
        action_mapper = components.action_mapper
        env = components.env
        cfg = PPOConfig(
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
    else:
        # Fallback runnable sanity-check with toy linear simulator.
        cfg = PPOConfig(
            obs_dim=16,
            theta_dim=4,
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
        simulator = ExampleLinearSimulator(obs_dim=cfg.obs_dim, action_dim=cfg.theta_dim, seed=cfg.seed)
        env = ExternalSimulatorEnv(
            simulator=simulator,
            max_steps=cfg.max_steps,
            reward_fn=example_reward_fn,
            action_mapper=identity_action_mapper,
        )
        action_mapper = identity_action_mapper

    # Train PPO and print a compact final metric.
    model, history = train_ppo(env, cfg)
    print("Training finished.")
    print(f"Final mean rollout reward: {history['mean_reward_rollout'][-1]:.6f}")

    # Optional post-training evaluation for transparent learned-vs-fixed comparison.
    if args.backend == "steane" and args.post_eval_episodes > 0:
        assert simulator is not None
        assert action_mapper is not None
        assert steane_cfg is not None

        device = torch.device(cfg.device)

        # Use separate evaluation simulators with drift reset per episode to ensure
        # learned-vs-fixed comparison is not biased by non-stationary drift phase.
        eval_cfg = replace(steane_cfg, reset_drift_on_episode=True)
        eval_sim_learned = SteaneOnlineSteeringSimulator(eval_cfg)
        eval_sim_fixed = SteaneOnlineSteeringSimulator(eval_cfg)

        def learned_policy(obs: np.ndarray) -> np.ndarray:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                theta = model.actor(obs_t).squeeze(0).cpu().numpy()
            return theta.astype(np.float32)

        def fixed_policy(_obs: np.ndarray) -> np.ndarray:
            return np.zeros(simulator.action_dim, dtype=np.float32)

        learned_metrics = evaluate_steane_policy_fn(
            simulator=eval_sim_learned,
            action_mapper=action_mapper,
            policy_fn=learned_policy,
            episodes=args.post_eval_episodes,
        )
        fixed_metrics = evaluate_steane_policy_fn(
            simulator=eval_sim_fixed,
            action_mapper=action_mapper,
            policy_fn=fixed_policy,
            episodes=args.post_eval_episodes,
        )

        dr_imp = (fixed_metrics["detector_rate"] - learned_metrics["detector_rate"]) / max(
            1e-12, fixed_metrics["detector_rate"]
        )
        ler_imp = (fixed_metrics["ler_proxy"] - learned_metrics["ler_proxy"]) / max(
            1e-12, fixed_metrics["ler_proxy"]
        )
        print("Post-train Steane evaluation (learned vs fixed-zero):")
        print(
            "  Learned: "
            f"DR={learned_metrics['detector_rate']:.6f}, "
            f"success={learned_metrics['success_rate']:.6f}, "
            f"LER~={learned_metrics['ler_proxy']:.6f}"
        )
        print(
            "  Fixed0 : "
            f"DR={fixed_metrics['detector_rate']:.6f}, "
            f"success={fixed_metrics['success_rate']:.6f}, "
            f"LER~={fixed_metrics['ler_proxy']:.6f}"
        )
        print(f"  Relative improvement: DR={dr_imp * 100.0:+.2f}%, LER~={ler_imp * 100.0:+.2f}%")


def parse_args() -> argparse.Namespace:
    """CLI options for switching backends and key hyperparameters.

    Examples:
      python -m rl_train.train --backend steane --steane-stepping-mode online_rounds
      python -m rl_train.train --backend toy --total-timesteps 5000
    """
    parser = argparse.ArgumentParser(description="Train PPO with Steane adapter or toy simulator.")

    # Common PPO/runtime options.
    parser.add_argument("--backend", choices=["steane", "toy"], default="steane")
    parser.add_argument("--code-family", choices=list(available_code_families()), default="steane")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=8_000)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-limit", type=float, default=2.0)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=128)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-hidden-dim", type=int, default=128)
    parser.add_argument("--ppo-use-layer-norm", action="store_true")
    parser.add_argument("--google-paper-ppo-preset", action="store_true")

    # Steane-adapter options.
    parser.add_argument(
        "--steane-stepping-mode",
        choices=["candidate_eval", "online_rounds"],
        default="candidate_eval",
    )
    parser.add_argument("--steane-n-rounds", type=int, default=25)
    parser.add_argument("--steane-shots-per-step", type=int, default=400)
    parser.add_argument(
        "--steane-control-mode",
        choices=["global", "gate_specific"],
        default="gate_specific",
    )
    parser.add_argument("--steane-control-dim", type=int, default=8)
    parser.add_argument("--steane-n-1q-control-slots", type=int, default=24)
    parser.add_argument("--steane-n-2q-control-slots", type=int, default=24)
    parser.add_argument("--steane-syndrome-mode", choices=["MV", "DE"], default="DE")
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
    parser.add_argument(
        "--steane-reward-mode",
        choices=["legacy_success", "paper_surrogate"],
        default="paper_surrogate",
    )
    parser.add_argument("--steane-action-penalty-coef", type=float, default=1e-3)
    parser.add_argument("--steane-miscal-penalty-coef", type=float, default=0.0)
    parser.add_argument("--steane-success-bonus-coef", type=float, default=0.0)
    parser.add_argument("--post-eval-episodes", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
