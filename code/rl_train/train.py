"""
Entry script for your custom simulator + PPO setup.

Replace TODO sections with your task definition.
"""

from typing import Dict

import numpy as np

from .config import PPOConfig
from .env import ExternalSimulatorEnv, identity_action_mapper, todo_reward_fn
from .example_simulator import ExampleLinearSimulator
from .interfaces import SimulatorTransition
from .ppo import train_ppo


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


def main() -> None:
    """
    Build config, instantiate environment, and run PPO training.

    The script contains two mutually exclusive setup paths:
      - Option A: real simulator integration (recommended for your task)
      - Option B: toy simulator sanity check (runs once torch is installed)
    """
    # User-facing run configuration.
    cfg = PPOConfig(
        obs_dim=16,       # TODO: set your observation size
        theta_dim=4,      # TODO: set number of policy parameters theta
        max_steps=50,     # user-definable episode horizon
        total_timesteps=40_000,
        rollout_steps=256,
        device="cpu",
    )

    # ------------------------------------------------------------
    # Option A (recommended now): plug your real simulator
    # simulator = YourSimulator()
    # env = ExternalSimulatorEnv(
    #     simulator=simulator,
    #     max_steps=cfg.max_steps,
    #     reward_fn=todo_reward_fn,            # TODO: replace with your formula
    #     action_mapper=identity_action_mapper, # TODO: replace with your external a(theta) mapper
    #     # terminate_fn=...                    # optional: task-specific early stop
    # )
    # ------------------------------------------------------------

    # Option B (runnable sanity check): use toy simulator.
    simulator = ExampleLinearSimulator(obs_dim=cfg.obs_dim, action_dim=cfg.theta_dim, seed=cfg.seed)
    env = ExternalSimulatorEnv(
        simulator=simulator,
        max_steps=cfg.max_steps,
        reward_fn=example_reward_fn,
        action_mapper=identity_action_mapper,
    )

    # Train PPO and print a compact final metric.
    _, history = train_ppo(env, cfg)
    print("Training finished.")
    print(f"Final mean rollout reward: {history['mean_reward_rollout'][-1]:.6f}")


if __name__ == "__main__":
    main()
