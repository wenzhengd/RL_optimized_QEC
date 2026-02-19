"""Subtask-2: Train reinforcement learning policy.

Important: this subtask stays quantum-free internally.
It only consumes dataset files produced by subtask1.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Define command line arguments for RL training."""
    parser = argparse.ArgumentParser(description="Train RL policy (skeleton).")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load dataset, train a policy, and save policy + metrics.
    print("[Skeleton] subtask2_rl/train_rl.py")
    print("Received args:", args)


if __name__ == "__main__":
    main()
