"""Subtask-1: class-based Stim simulation entrypoint.

Why class-based:
- easier to add multiple code backends over time
- keeps schedule parsing/validation reusable
- keeps CLI thin while simulation logic grows
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import stim  # type: ignore
except Exception:  # pragma: no cover - optional during scaffold stage
    stim = None


@dataclass
class SimConfig:
    """Configuration object for one simulation run."""

    code: str
    rounds: int
    shots: int
    seed: int
    schedule_file: str
    out: str
    distance: int = 3
    save_measurements: bool = False
    strict: bool = False


def parse_args() -> argparse.Namespace:
    """Define command line arguments for Stim simulation."""
    parser = argparse.ArgumentParser(description="Run Stim QEC simulation.")
    parser.add_argument(
        "--code",
        required=True,
        choices=["small_surface", "steane7", "shor9", "gauge_color_15"],
    )
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--schedule_file", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--save_measurements", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


class StimQECSimulator:
    """Class wrapper for QEC simulation with time-dependent Pauli schedule."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def run(self) -> Path:
        """Run the full pipeline: load schedule -> simulate -> save output."""
        self._validate_basic_args()
        schedule, schedule_text = self._load_schedule()
        self._validate_schedule(schedule)

        detector_history, backend_mode = self._simulate_detector_history(schedule)
        measurement_history = None
        if self.config.save_measurements:
            measurement_history = self._simulate_measurements_stub(detector_history)

        meta = self._build_metadata(
            schedule=schedule,
            schedule_text=schedule_text,
            detector_history=detector_history,
            backend_mode=backend_mode,
        )
        out_path = self._resolve_output_path()
        self._save_result(out_path, detector_history, meta, measurement_history)
        return out_path

    def _validate_basic_args(self) -> None:
        if self.config.rounds < 1:
            raise ValueError("`--rounds` must be >= 1.")
        if self.config.shots < 1:
            raise ValueError("`--shots` must be >= 1.")
        if self.config.distance < 2:
            raise ValueError("`--distance` must be >= 2.")

    def _load_schedule(self) -> tuple[dict[str, Any], str]:
        """Load round-dependent Pauli schedule from JSON."""
        path = Path(self.config.schedule_file)
        text = path.read_text()
        schedule = json.loads(text)
        return schedule, text

    def _validate_schedule(self, schedule: dict[str, Any]) -> None:
        """Validate probabilities and schedule-round consistency."""
        if schedule.get("schedule_type") != "per_round":
            msg = "schedule_type must be 'per_round'."
            if self.config.strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}")

        declared_rounds = schedule.get("rounds")
        if declared_rounds is not None and int(declared_rounds) != self.config.rounds:
            msg = (
                f"Schedule rounds ({declared_rounds}) != --rounds ({self.config.rounds})."
            )
            if self.config.strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}")

        default = schedule.get("default", {})
        self._validate_prob_triplet(default, label="default")
        overrides = schedule.get("overrides", {})
        for k, probs in overrides.items():
            self._validate_prob_triplet(probs, label=f"override round {k}")

    @staticmethod
    def _validate_prob_triplet(probs: dict[str, Any], label: str) -> None:
        px = float(probs.get("p_x", 0.0))
        py = float(probs.get("p_y", 0.0))
        pz = float(probs.get("p_z", 0.0))
        if min(px, py, pz) < 0:
            raise ValueError(f"{label}: probabilities must be >= 0.")
        if px + py + pz > 1.0:
            raise ValueError(f"{label}: require p_x + p_y + p_z <= 1.")

    def _round_probs(self, schedule: dict[str, Any], r: int) -> tuple[float, float, float]:
        """Return (p_x, p_y, p_z) for round index r."""
        default = schedule.get("default", {})
        probs = dict(default)
        overrides = schedule.get("overrides", {})
        if str(r) in overrides:
            probs.update(overrides[str(r)])
        px = float(probs.get("p_x", 0.0))
        py = float(probs.get("p_y", 0.0))
        pz = float(probs.get("p_z", 0.0))
        return px, py, pz

    def _detectors_per_round(self) -> int:
        """Choose detector count per round by code type.

        This is a scaffold estimate. Replace with exact count from real Stim circuit
        once backend construction is implemented.
        """
        if self.config.code == "small_surface":
            d = self.config.distance
            return max(1, 2 * d * (d - 1))
        if self.config.code == "steane7":
            return 6
        if self.config.code == "shor9":
            return 8
        if self.config.code == "gauge_color_15":
            return 14
        raise ValueError(f"Unsupported code: {self.config.code}")

    def _simulate_detector_history(
        self, schedule: dict[str, Any]
    ) -> tuple[np.ndarray, str]:
        """Generate detector history.

        Behavior:
        - If Stim is available and code is small_surface: use round-wise Stim sampling.
        - Otherwise: use deterministic scaffold sampling.
        """
        if stim is not None and self.config.code == "small_surface":
            try:
                hist = self._simulate_detector_history_stim_small_surface(schedule)
                return hist, "stim_roundwise_small_surface"
            except Exception as exc:
                print(f"[WARN] Stim backend failed; fallback to scaffold. reason={exc}")

        hist = self._simulate_detector_history_scaffold(schedule)
        return hist, "scaffold"

    def _simulate_detector_history_scaffold(
        self, schedule: dict[str, Any]
    ) -> np.ndarray:
        """Fallback history generator used when Stim backend is unavailable."""
        n_det = self._detectors_per_round()
        hist = np.zeros((self.config.shots, self.config.rounds, n_det), dtype=np.uint8)

        if stim is None:
            print("[WARN] stim not available; using scaffold detector sampling.")

        for r in range(self.config.rounds):
            px, py, pz = self._round_probs(schedule, r)
            p_total = min(1.0, max(0.0, px + py + pz))
            # Simple monotonic mapping for scaffold mode.
            p_trigger = min(0.5, 0.02 + 2.0 * p_total)
            draws = self.rng.random((self.config.shots, n_det)) < p_trigger
            hist[:, r, :] = draws.astype(np.uint8)
        return hist

    def _simulate_detector_history_stim_small_surface(
        self, schedule: dict[str, Any]
    ) -> np.ndarray:
        """Round-wise Stim sampling for small_surface.

        Note:
        - This is a practical bridge implementation.
        - Each round is sampled with its own one-round circuit and per-round Pauli rate.
        """
        all_rounds: list[np.ndarray] = []
        n_det_ref: int | None = None

        for r in range(self.config.rounds):
            px, py, pz = self._round_probs(schedule, r)
            p_total = min(1.0, max(0.0, px + py + pz))

            circuit = self._build_small_surface_round_circuit(p_total)
            sampler = circuit.compile_detector_sampler(seed=self.config.seed + r)
            det = sampler.sample(self.config.shots, append_observables=False)
            det = np.asarray(det, dtype=np.uint8)

            if det.ndim != 2:
                raise RuntimeError(f"Unexpected detector sample shape at round {r}: {det.shape}")
            if n_det_ref is None:
                n_det_ref = det.shape[1]
            elif det.shape[1] != n_det_ref:
                raise RuntimeError(
                    f"Detector count changed across rounds: {n_det_ref} vs {det.shape[1]}"
                )

            all_rounds.append(det)

        # Stack into (shots, rounds, n_detectors_per_round).
        hist = np.stack(all_rounds, axis=1)
        return hist

    def _build_small_surface_round_circuit(self, p_total: float) -> Any:
        """Build one-round small surface code circuit with effective Pauli noise."""
        # Prefer a richer noise model if this Stim version supports all kwargs.
        try:
            return stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=self.config.distance,
                rounds=1,
                after_clifford_depolarization=p_total,
                before_round_data_depolarization=p_total,
                before_measure_flip_probability=0.0,
                after_reset_flip_probability=0.0,
            )
        except TypeError:
            # Fallback for older Stim versions with fewer generator kwargs.
            return stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=self.config.distance,
                rounds=1,
                after_clifford_depolarization=p_total,
            )

    def _simulate_measurements_stub(self, detector_history: np.ndarray) -> np.ndarray:
        """Create placeholder measurement history when user requests it."""
        shots, rounds, n_det = detector_history.shape
        total = rounds * n_det
        return detector_history.reshape(shots, total).copy()

    def _build_metadata(
        self,
        schedule: dict[str, Any],
        schedule_text: str,
        detector_history: np.ndarray,
        backend_mode: str,
    ) -> dict[str, Any]:
        shots, rounds, n_det = detector_history.shape
        digest = hashlib.sha256(schedule_text.encode("utf-8")).hexdigest()
        stim_version = getattr(stim, "__version__", None) if stim is not None else None
        return {
            "code": self.config.code,
            "rounds": self.config.rounds,
            "shots": self.config.shots,
            "seed": self.config.seed,
            "distance": self.config.distance,
            "schedule_file": self.config.schedule_file,
            "schedule_digest": digest,
            "n_detectors_total": int(rounds * n_det),
            "n_detectors_per_round": int(n_det),
            "schedule_type": schedule.get("schedule_type"),
            "stim_version": stim_version,
            "backend_mode": backend_mode,
        }

    def _resolve_output_path(self) -> Path:
        if self.config.out:
            return Path(self.config.out)
        return Path("subtask1_stim_qec") / f"stim_{self.config.code}_seed{self.config.seed}.npz"

    def _save_result(
        self,
        out_path: Path,
        detector_history: np.ndarray,
        meta: dict[str, Any],
        measurement_history: np.ndarray | None,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "detector_history": detector_history.astype(np.uint8),
            "time_round": np.arange(self.config.rounds, dtype=np.int32),
            "meta_json": json.dumps(meta),
        }
        if measurement_history is not None:
            kwargs["measurement_history"] = measurement_history.astype(np.uint8)
        np.savez_compressed(out_path, **kwargs)


def main() -> None:
    args = parse_args()
    config = SimConfig(
        code=args.code,
        rounds=args.rounds,
        shots=args.shots,
        seed=args.seed,
        schedule_file=args.schedule_file,
        out=args.out,
        distance=args.distance,
        save_measurements=args.save_measurements,
        strict=args.strict,
    )
    sim = StimQECSimulator(config)
    out_path = sim.run()
    print(f"[OK] Saved simulation file: {out_path}")


if __name__ == "__main__":
    main()
