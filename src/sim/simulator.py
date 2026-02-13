"""Simulator interface stub.

This module maps control input and latent context to a readout/aux pair.
It intentionally does not compute reward or termination.
"""

from __future__ import annotations

from typing import Any

from src.utils.types import ArrayLike, HistoryLike, InfoDict


class Simulator:
    """Minimal simulator core interface with deterministic placeholder output."""

    def simulate_step(
        self,
        step_index: int,
        action: ArrayLike,
        latent: Any,
        history: HistoryLike,
        rng: Any | None = None,
    ) -> tuple[ArrayLike, InfoDict]:
        """Simulate one step and return readout and aux diagnostics.

        Placeholder behavior returns a deterministic scalar readout 0.0.
        """

        _ = (step_index, action, latent, history, rng)
        readout = 0.0
        aux: InfoDict = {"simulator": "stub"}
        return readout, aux
