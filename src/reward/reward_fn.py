"""Pluggable reward interface stub.

Supports step-wise and terminal reward hooks without committing to a style.
"""

from __future__ import annotations

from src.utils.types import EpisodeRecord, HistoryLike, InfoDict, RewardMode


class RewardFunction:
    """Minimal reward functional interface with zero-valued placeholders."""

    def __init__(self, reward_mode: RewardMode = "step") -> None:
        self.reward_mode = reward_mode

    def compute_step(
        self,
        step_index: int,
        history: HistoryLike,
        latest_readout: object,
        aux: InfoDict,
    ) -> float | None:
        """Return per-step reward r_i (optional by task configuration)."""

        _ = (step_index, history, latest_readout, aux)
        return 0.0

    def compute_terminal(self, episode_record: EpisodeRecord) -> float | None:
        """Return terminal reward R for the full episode trajectory."""

        _ = episode_record
        return 0.0
