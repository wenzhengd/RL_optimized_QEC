"""Subtask-3 helper: plot benchmark comparison figures."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark comparisons (skeleton).")
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: read benchmark table and save comparison figures.
    print("[Skeleton] subtask3_testbed/plot_compare.py")
    print("Received args:", args)


if __name__ == "__main__":
    main()
