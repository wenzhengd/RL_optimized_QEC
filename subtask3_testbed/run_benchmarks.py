"""Subtask-3: benchmark policies across QEC codes and noise settings."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark matrix (skeleton).")
    parser.add_argument("--codes", type=str, required=True, help="Comma-separated list")
    parser.add_argument("--noise_files", type=str, required=True, help="Comma-separated list")
    parser.add_argument("--policies", type=str, required=True, help="Comma-separated list")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: iterate code/noise/policy combinations and save benchmark CSV.
    print("[Skeleton] subtask3_testbed/run_benchmarks.py")
    print("Received args:", args)


if __name__ == "__main__":
    main()
