"""Observation builder interface stub.

Builds policy-facing observations from rollout history and latest simulator outputs.
"""

from __future__ import annotations

from src.utils.types import HistoryLike, ObservationLike, InfoDict


class ObservationBuilder:
    """History-aware observation builder.

    Default stub includes control history so POMDP/history usage is supported.
    """

    def build(
        self,
        step_index: int,
        history: HistoryLike,
        latest_readout: object,
        aux: InfoDict,
    ) -> ObservationLike:
        """Construct an observation object for the current step."""

        actions = history.get("actions", [])
        return {
            "step_index": step_index,
            "action_history": actions,
            "latest_readout": latest_readout,
            "aux": aux,
        }
