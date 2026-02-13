"""Termination checker interface stub.

Supports hard cap (n_max) and optional early-stop threshold hooks.
"""

from __future__ import annotations

from src.utils.types import HistoryLike, InfoDict


class TerminationChecker:
    """Minimal termination policy.

    Args:
        n_max: Hard maximum step count for an episode.
        early_stop_threshold: Optional threshold hook on latest readout.
    """

    def __init__(self, n_max: int, early_stop_threshold: float | None = None) -> None:
        self.n_max = n_max
        self.early_stop_threshold = early_stop_threshold

    def check(
        self,
        step_index: int,
        history: HistoryLike,
        latest_readout: object,
        aux: InfoDict,
    ) -> tuple[bool, str]:
        """Return (done, reason) for this step."""

        _ = (history, aux)
        if step_index >= self.n_max:
            return True, "n_max_reached"

        if self.early_stop_threshold is not None:
            try:
                if float(latest_readout) >= self.early_stop_threshold:
                    return True, "early_stop_threshold"
            except (TypeError, ValueError):
                # Non-scalar readouts keep early-stop disabled in this stub.
                pass

        return False, ""
