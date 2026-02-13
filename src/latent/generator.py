"""Latent-process generator interface with minimal stubs.

This module provides an opaque latent source for episodic simulation.
No assumptions are made about latent-process dimensionality or physics.
"""

from __future__ import annotations

from typing import Any

from src.utils.types import InfoDict


class LatentGenerator:
    """Stochastic latent-process generator interface.

    Args:
        seed: Optional RNG seed hook for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def reset(self) -> Any:
        """Sample/reset latent state for a new episode.

        Returns:
            Opaque latent state/trajectory object.
        """

        return {"seed": self.seed}

    def step(self, step_index: int, latent_state: Any) -> Any:
        """Optional latent evolution hook.

        Placeholder behavior returns the input state unchanged.
        """

        _ = step_index
        return latent_state

    def get_rng_state(self) -> InfoDict:
        """Expose minimal RNG metadata hook for debugging."""

        return {"seed": self.seed}
