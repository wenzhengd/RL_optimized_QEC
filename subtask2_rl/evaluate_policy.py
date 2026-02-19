"""Subtask-2 helper: evaluate a trained policy on a dataset."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RL policy (skeleton).")
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load policy + dataset and compute evaluation metrics.
    print("[Skeleton] subtask2_rl/evaluate_policy.py")
    print("Received args:", args)


if __name__ == "__main__":
    main()
