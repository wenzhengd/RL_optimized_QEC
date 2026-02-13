"""Shared lightweight rollout record types.

These records are intentionally generic and avoid assumptions about:
- control dimensionality
- readout shape
- observation Markovity
- reward style (step-wise vs terminal-only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ArrayLike = Any
ObservationLike = Any
InfoDict = dict[str, Any]
MetadataDict = dict[str, Any]
HistoryLike = dict[str, Any]
RewardMode = Literal["step", "terminal", "combined"]


@dataclass
class StepRecord:
    """Single environment transition record.

    Attributes:
        step_index: 1-based step index i within the episode.
        action: Control action theta_i (continuous, possibly high-dimensional).
        readout: Simulator readout C_i (scalar or vector, task-dependent).
        observation: Observation o_i returned to the policy (can be history-based).
        reward: Per-step reward r_i if used; None for terminal-only reward mode.
        done: Episode termination flag after this step.
        info: Auxiliary diagnostics for debugging/analysis.
    """

    step_index: int
    action: ArrayLike
    readout: ArrayLike
    observation: ObservationLike
    reward: float | None
    done: bool
    info: InfoDict = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    """Aggregated rollout record for one episode.

    Attributes:
        episode_id: Unique episode identifier (int/str depending on caller).
        n_max: Maximum allowed episode length (hard cap).
        length: Actual executed step count T (<= n_max with early termination support).
        actions: Ordered sequence of actions {theta_i}.
        readouts: Ordered sequence of readouts {C_i}.
        observations: Ordered sequence of observations {o_i}.
        rewards: Ordered per-step rewards {r_i}; entries can be None in terminal-only mode.
        terminal_reward: Optional terminal reward R computed at episode end.
        termination_reason: Reason label for stop event (cap, threshold, etc.).
        latent_metadata: Optional latent-process metadata for debugging only.
        mc_metadata: Optional Monte Carlo metadata for debugging only.
        steps: Optional per-step structured records for full transition detail.
    """

    episode_id: int | str
    n_max: int
    length: int = 0
    actions: list[ArrayLike] = field(default_factory=list)
    readouts: list[ArrayLike] = field(default_factory=list)
    observations: list[ObservationLike] = field(default_factory=list)
    rewards: list[float | None] = field(default_factory=list)
    terminal_reward: float | None = None
    termination_reason: str = ""
    latent_metadata: MetadataDict = field(default_factory=dict)
    mc_metadata: MetadataDict = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)

    def add_step(self, step: StepRecord) -> None:
        """Append one step and keep aggregate sequences synchronized."""

        self.steps.append(step)
        self.actions.append(step.action)
        self.readouts.append(step.readout)
        self.observations.append(step.observation)
        self.rewards.append(step.reward)
        self.length = len(self.steps)
