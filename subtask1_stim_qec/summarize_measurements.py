"""Subtask-1: summarize detector history from Stim-QEC simulation output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Summarize Stim-QEC detector history.")
    parser.add_argument("--input", type=str, required=True, help="Input .npz file from simulate_stim.py")
    parser.add_argument("--out_json", type=str, default="", help="Optional output summary JSON path")
    parser.add_argument("--save_plot", action="store_true", help="Save trigger-rate plot")
    return parser.parse_args()


def _parse_meta(meta_json_obj: Any) -> dict[str, Any]:
    """Parse JSON metadata saved inside npz."""
    if isinstance(meta_json_obj, np.ndarray):
        raw = meta_json_obj.item()
    else:
        raw = meta_json_obj
    return json.loads(raw)


def _default_out_json(code: str, seed: int) -> Path:
    _ = seed
    return Path("subtask1_stim_qec") / "subtask1_data" / f"summary_{code}.json"


def _default_plot_path(code: str, seed: int) -> Path:
    _ = seed
    return Path("subtask1_stim_qec") / "subtask1_data" / f"trigger_rate_{code}.png"


def _round_rate_autocorr(round_rates: np.ndarray) -> list[float]:
    """Compute a short normalized autocorrelation of the per-round trigger series."""
    x = np.asarray(round_rates, dtype=np.float64)
    if x.size <= 1:
        return [1.0]
    x = x - np.mean(x)
    var = float(np.mean(x**2))
    if var <= 1e-15:
        return [1.0] + [0.0] * min(4, x.size - 1)
    max_lag = min(5, x.size)
    acf = [1.0]
    for lag in range(1, max_lag):
        acf.append(float(np.mean(x[:-lag] * x[lag:]) / var))
    return acf


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)

    data = np.load(in_path, allow_pickle=True)
    if "detector_history" not in data:
        raise KeyError("Input file is missing `detector_history`.")

    detector_history = np.asarray(data["detector_history"], dtype=np.uint8)
    if detector_history.ndim != 3:
        raise ValueError(
            f"`detector_history` must be rank-3 (shots, rounds, detectors). Got shape={detector_history.shape}."
        )

    shots, rounds, n_detectors = detector_history.shape
    # Binary consistency check required by spec.
    unique_vals = np.unique(detector_history)
    is_binary = bool(np.all(np.isin(unique_vals, [0, 1])))

    meta: dict[str, Any] = {}
    if "meta_json" in data:
        meta = _parse_meta(data["meta_json"])
    code = str(meta.get("code", "unknown_code"))
    seed = int(meta.get("seed", -1))

    # Per-round trigger rate = mean over shots and detectors for each round.
    per_round_trigger_rate = detector_history.mean(axis=(0, 2)).astype(float)
    global_trigger_rate = float(detector_history.mean())
    temporal_acf = _round_rate_autocorr(per_round_trigger_rate)

    summary = {
        "input_file": str(in_path),
        "code": code,
        "seed": seed,
        "shape": {
            "shots": int(shots),
            "rounds": int(rounds),
            "n_detectors_per_round": int(n_detectors),
        },
        "binary_check": {
            "is_binary": is_binary,
            "unique_values": unique_vals.tolist(),
        },
        "global_trigger_rate": global_trigger_rate,
        "per_round_trigger_rate": per_round_trigger_rate.tolist(),
        "temporal_autocorr_round_rate_lag0_to_lag4": temporal_acf,
    }

    print("[Measurement Summary]")
    print(json.dumps(summary, indent=2))

    out_json = Path(args.out_json) if args.out_json else _default_out_json(code, seed)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[OK] Saved summary JSON: {out_json}")

    if args.save_plot:
        plot_path = _default_plot_path(code, seed)
        x = np.arange(rounds)
        plt.figure(figsize=(7, 4))
        plt.plot(x, per_round_trigger_rate, marker="o", label="Per-round trigger rate")
        plt.xlabel("Round index")
        plt.ylabel("Trigger rate")
        plt.title(f"Detector Trigger Rate vs Round ({code})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=140)
        plt.close()
        print(f"[OK] Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
