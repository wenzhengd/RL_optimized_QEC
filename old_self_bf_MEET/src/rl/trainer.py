"""Training loop skeleton without policy updates.

This module orchestrates rollout collection and metric reporting only.
No learning/update logic is implemented at this stage.
"""

from __future__ import annotations

from typing import Any

from src.rl.rollout import collect_rollouts
from src.utils.types import EpisodeRecord


def training_loop(env: Any, policy: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Run a minimal training loop skeleton.

    Args:
        env: Environment compatible with rollout collection.
        policy: Policy compatible with rollout collection.
        config: Training settings. Supported keys:
            - epochs (int)
            - batches_per_epoch (int)
            - batch_size (int)
            - max_steps (int)

    Returns:
        Dict containing epoch-level metrics without any RL update outputs.
    """

    epochs = int(config.get("epochs", 1))
    batches_per_epoch = int(config.get("batches_per_epoch", 1))
    batch_size = int(config.get("batch_size", 1))
    max_steps = int(config.get("max_steps", 1))

    history: dict[str, list[float | int]] = {
        "episode_lengths": [],
        "episode_reward_sums": [],
        "epoch_mean_length": [],
        "epoch_mean_reward_sum": [],
    }

    for epoch in range(1, epochs + 1):
        epoch_lengths: list[int] = []
        epoch_reward_sums: list[float] = []

        for _ in range(batches_per_epoch):
            episodes: list[EpisodeRecord] = collect_rollouts(
                policy=policy,
                env=env,
                batch_size=batch_size,
                max_steps=max_steps,
            )

            for episode in episodes:
                length = int(episode.length)
                reward_sum = float(
                    sum(reward for reward in episode.rewards if reward is not None)
                )
                history["episode_lengths"].append(length)
                history["episode_reward_sums"].append(reward_sum)
                epoch_lengths.append(length)
                epoch_reward_sums.append(reward_sum)

        mean_length = (
            sum(epoch_lengths) / len(epoch_lengths) if epoch_lengths else 0.0
        )
        mean_reward = (
            sum(epoch_reward_sums) / len(epoch_reward_sums)
            if epoch_reward_sums
            else 0.0
        )
        history["epoch_mean_length"].append(mean_length)
        history["epoch_mean_reward_sum"].append(mean_reward)

        print(
            f"epoch={epoch} "
            f"mean_length={mean_length:.3f} "
            f"mean_reward_sum={mean_reward:.3f}"
        )

    return history
