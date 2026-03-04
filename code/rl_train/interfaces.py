"""Shared interfaces and callback signatures for simulator-driven RL."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

import numpy as np


@dataclass
class SimulatorTransition:
    """
    Single transition record returned by external simulator step().

    Attributes:
        next_obs: Observation vector/state visible to the policy at next step.
        info: Arbitrary simulator-side diagnostics used by reward/termination logic.
        done: Optional simulator-native terminal flag.
    """

    next_obs: np.ndarray
    info: Dict[str, Any]
    done: bool = False


class SimulatorProtocol(Protocol):
    """
    Minimal contract that any callable simulator must satisfy.

    The scaffold keeps this interface intentionally small:
      - reset() initializes one episode and returns first observation.
      - step(action) advances one step and returns a SimulatorTransition.
    """

    def reset(self) -> np.ndarray:
        """Reset simulator state and return initial observation."""
        ...

    def step(self, action: np.ndarray) -> SimulatorTransition:
        """Apply action and return simulator transition."""
        ...


# theta_t: parameters output by policy network (to be mapped to system action externally).
ActionMapper = Callable[[np.ndarray], np.ndarray]

# r_t = reward_fn(obs_t, theta_t, action_t, obs_{t+1}, simulator_feedback, t+1)
RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], int], float]

# task-specific terminal condition; max_steps is handled separately.
TerminateFn = Callable[[np.ndarray, Dict[str, Any], int], bool]
