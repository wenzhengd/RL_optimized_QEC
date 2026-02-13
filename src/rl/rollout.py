"""Rollout collection utilities for policy-environment interaction."""

from __future__ import annotations

from typing import Any

from src.utils.types import EpisodeRecord


def collect_rollouts(
    policy: Any,
    env: Any,
    batch_size: int,
    max_steps: int,
) -> list[EpisodeRecord]:
    """Collect a batch of rollouts without performing any RL updates.

    Args:
        policy: Policy object exposing ``act(observation, deterministic=False)``.
        env: Environment object exposing ``reset()`` and ``step(action)`` and
            providing ``current_episode`` as an ``EpisodeRecord`` after rollout.
        batch_size: Number of trajectories to collect.
        max_steps: Per-trajectory step cap for collection safety.

    Returns:
        A list of ``EpisodeRecord`` objects, one per trajectory.
    """

    episodes: list[EpisodeRecord] = []

    for _ in range(batch_size):
        observation = env.reset()
        if hasattr(policy, "reset_hidden_state"):
            policy.reset_hidden_state()

        done = False
        step_count = 0
        while not done and step_count < max_steps:
            action = policy.act(observation)
            observation, _reward, done, _info = env.step(action)
            step_count += 1

        episode = getattr(env, "current_episode", None)
        if not isinstance(episode, EpisodeRecord):
            raise RuntimeError(
                "Environment did not expose a valid EpisodeRecord via current_episode."
            )
        episodes.append(episode)

    return episodes
