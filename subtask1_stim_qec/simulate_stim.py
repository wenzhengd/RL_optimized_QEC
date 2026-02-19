"""Subtask-1: Stim-based stabilizer measurement simulation.

This is a minimal skeleton with comments for future implementation.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Define command line arguments for Stim simulation."""
    parser = argparse.ArgumentParser(description="Run Stim QEC simulation (skeleton).")
    parser.add_argument("--code", required=True, choices=["repetition", "surface_d3", "surface_d5"])
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--noise_file", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load noise_file from subtask0 and run Stim circuit simulation.
    # TODO: save measurements/detectors/observables into .npz.
    print("[Skeleton] subtask1_stim_qec/simulate_stim.py")
    print("Received args:", args)


if __name__ == "__main__":
    main()
