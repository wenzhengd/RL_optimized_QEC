"""Environment wrapper that bridges policy outputs and external simulator calls."""

from typing import Dict, Optional, Tuple

import numpy as np

from .interfaces import ActionMapper, RewardFn, SimulatorProtocol, TerminateFn


def identity_action_mapper(theta: np.ndarray) -> np.ndarray:
    """
    Default action mapper.

    This leaves the policy output unchanged:
      a_t = theta_t

    Replace this when your control pipeline needs nonlinear transforms,
    clamping, quantization, hardware formatting, or parameter expansion.
    """
    return theta


def todo_reward_fn(
    obs_t: np.ndarray,
    theta_t: np.ndarray,
    action_t: np.ndarray,
    obs_tp1: np.ndarray,
    simulator_feedback: Dict,
    timestep: int,
) -> float:
    """
    TODO: Replace this with your reward formula.

    Args:
        obs_t: Observation at time t.
        theta_t: Policy output parameters at time t.
        action_t: Mapped action applied to simulator.
        obs_tp1: Observation at time t+1 after action.
        simulator_feedback: Extra diagnostics returned by simulator.
        timestep: Current step index (1-based within an episode).

    Returns:
        Scalar reward r_t.

    Example:
        reward = metric(obs_tp1, simulator_feedback) - metric(obs_t, simulator_feedback)
    """
    _ = (obs_t, theta_t, action_t, obs_tp1, simulator_feedback, timestep)
    return 0.0


def default_terminate_fn(obs_t: np.ndarray, simulator_feedback: Dict, timestep: int) -> bool:
    """
    Default task-specific terminal condition.

    Notes:
        - This callback handles only task-level early termination.
        - Episode horizon termination (`t >= max_steps`) is enforced separately.
    """
    _ = (obs_t, simulator_feedback, timestep)
    return False


class ExternalSimulatorEnv:
    """
    Thin RL environment wrapper for a callable external simulator.

    Time is discrete: t_1 -> t_2 -> ... -> t_N.
    At each step:
      1) policy outputs theta_t
      2) action_mapper maps theta_t -> system action a_t
      3) simulator executes a_t and returns feedback
      4) reward_fn computes r_t

    The purpose of this class is to isolate domain-specific pieces
    (simulator/action mapping/reward/termination) from PPO internals.
    """

    def __init__(
        self,
        simulator: SimulatorProtocol,
        max_steps: int,
        reward_fn: RewardFn,
        action_mapper: Optional[ActionMapper] = None,
        terminate_fn: Optional[TerminateFn] = None,
    ) -> None:
        """
        Initialize wrapper and callbacks.

        Args:
            simulator: Callable simulator implementing reset()/step().
            max_steps: Per-episode hard step limit.
            reward_fn: Reward callback used after every simulator step.
            action_mapper: Optional callback mapping theta_t to action_t.
            terminate_fn: Optional task-specific early termination callback.
        """
        self.simulator = simulator
        self.max_steps = max_steps
        self.reward_fn = reward_fn
        self.action_mapper = action_mapper or identity_action_mapper
        self.terminate_fn = terminate_fn or default_terminate_fn

        self.t = 0
        self._last_obs: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        """
        Reset episode state and return initial observation.

        Returns:
            Initial observation vector as float32 NumPy array.
        """
        # Reset per-episode time index before calling simulator.
        self.t = 0
        # Normalize dtype at wrapper boundary so policy always sees float32.
        self._last_obs = np.asarray(self.simulator.reset(), dtype=np.float32)
        # Return a copy to avoid accidental in-place mutations by caller.
        return self._last_obs.copy()

    def step(self, theta_t: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Run one environment transition using policy output parameters.

        Args:
            theta_t: Policy output parameters for current step.

        Returns:
            Tuple(next_obs, reward, done, info)
        """
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step().")

        # Standardize policy output to float32 and map to simulator action space.
        theta_t = np.asarray(theta_t, dtype=np.float32)
        action_t = np.asarray(self.action_mapper(theta_t), dtype=np.float32)

        # Delegate true dynamics to external simulator.
        transition = self.simulator.step(action_t)
        obs_tp1 = np.asarray(transition.next_obs, dtype=np.float32)

        # Time index is 1-based for reward/termination callbacks.
        self.t += 1
        # Reward is defined externally so task logic stays outside PPO core.
        reward_t = float(
            self.reward_fn(
                self._last_obs,
                theta_t,
                action_t,
                obs_tp1,
                transition.info,
                self.t,
            )
        )

        # Combine three terminal sources:
        # 1) hard horizon,
        # 2) task-specific early stop callback,
        # 3) simulator-native done flag.
        done_max_steps = self.t >= self.max_steps
        done_task = bool(self.terminate_fn(obs_tp1, transition.info, self.t))
        done = bool(transition.done or done_task or done_max_steps)

        # Cache observation for next reward computation.
        self._last_obs = obs_tp1
        # Return defensive copies for safety.
        return obs_tp1.copy(), reward_t, done, dict(transition.info)
