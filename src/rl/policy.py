"""RL policy interface stubs for continuous, high-dimensional actions.

This module defines a minimal policy contract without committing to a
specific learning algorithm or neural network architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PolicyInterface(ABC):
    """Abstract policy interface for action selection.

    Implementations consume an observation ``o_i`` and produce a continuous
    action ``theta_i`` that can be high-dimensional.
    """

    @abstractmethod
    def act(self, observation: Any, deterministic: bool = False) -> list[float]:
        """Return a continuous action for the given observation."""

    @abstractmethod
    def reset_hidden_state(self) -> None:
        """Reset recurrent hidden state at episode boundaries."""

    def get_hidden_state(self) -> Any | None:
        """Return current hidden state if the policy is recurrent.

        Non-recurrent implementations may return ``None``.
        """

        return None


class DummyContinuousPolicy(PolicyInterface):
    """Minimal import-safe policy stub that returns zero actions.

    Args:
        action_dim: Number of continuous action components to return.
    """

    def __init__(self, action_dim: int = 1) -> None:
        self.action_dim = max(1, int(action_dim))
        self._hidden_state: Any | None = None

    def act(self, observation: Any, deterministic: bool = False) -> list[float]:
        """Return a dummy continuous action vector with shape ``[action_dim]``."""

        _ = (observation, deterministic)
        return [0.0] * self.action_dim

    def reset_hidden_state(self) -> None:
        """Reset hidden state placeholder for future recurrent variants."""

        self._hidden_state = None

    def get_hidden_state(self) -> Any | None:
        """Expose hidden state placeholder for recurrent-policy compatibility."""

        return self._hidden_state
