"""Toy simulator used to smoke-test the PPO scaffold end-to-end."""

from typing import Dict

import numpy as np

from .interfaces import SimulatorTransition


class ExampleLinearSimulator:
    """
    Minimal simulator to verify the training pipeline end-to-end.

    Dynamics:
      state_{t+1} = 0.95 * state_t + B @ action_t
    """

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 42) -> None:
        """
        Initialize linear dynamics and target.

        Args:
            obs_dim: Observation/state vector dimension.
            action_dim: Action vector dimension.
            seed: Random seed for deterministic reproducibility.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

        # B controls how action components influence state evolution.
        self.B = self.rng.normal(0.0, 0.05, size=(obs_dim, action_dim)).astype(np.float32)

        # Internal state and fixed target used in example reward function.
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        self.target = np.ones(self.obs_dim, dtype=np.float32) * 0.2

    def reset(self) -> np.ndarray:
        """
        Reset simulator state for a new episode.

        Returns:
            Initial observation/state vector.
        """
        # Randomize initial state around zero so policy sees diverse starts.
        self.state = self.rng.normal(0.0, 0.2, size=self.obs_dim).astype(np.float32)
        return self.state.copy()

    def step(self, action: np.ndarray) -> SimulatorTransition:
        """
        Apply one action and advance dynamics by one time step.

        Args:
            action: Continuous action vector.

        Returns:
            SimulatorTransition containing next observation and diagnostics.
        """
        action = np.asarray(action, dtype=np.float32)

        # Simple stable linear update.
        self.state = (0.95 * self.state + self.B @ action).astype(np.float32)

        # Provide metric in info so reward callback can use it.
        info: Dict[str, float] = {
            "distance_to_target": float(np.linalg.norm(self.state - self.target)),
        }
        return SimulatorTransition(next_obs=self.state.copy(), info=info, done=False)
