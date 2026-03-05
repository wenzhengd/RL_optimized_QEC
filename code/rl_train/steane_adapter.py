"""Steane-code simulator adapter that matches the rl_train simulator protocol.

This module provides a bridge between:
  - `quantum_simulation.steane_code_simulator` (batch experiment API),
  - `rl_train.env.ExternalSimulatorEnv` (reset/step API).

Two stepping modes are supported:
  1) `candidate_eval`:
     each RL step evaluates one full memory experiment of `n_rounds`.
  2) `online_rounds`:
     each RL step evaluates exactly one round (6 stabilizer steps),
     and simulator sets `done=True` after `n_rounds` RL steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence

import numpy as np

from quantum_simulation.noise_engine import (
    GoogleLikeDepolarizingNoiseModel,
    GoogleLikeGateSpecificNoiseModel,
)
from quantum_simulation.steane_code_simulator import STABILIZER_SEQUENCE, SteaneQECSimulator

from .interfaces import SimulatorTransition


@dataclass
class SteaneAdapterConfig:
    """Configuration for RL-facing Steane simulation."""

    n_rounds: int = 10
    shots_per_step: int = 200
    initial_state: str = "+Z"
    meas_basis: str = "Z"
    syndrome_mode: Literal["MV", "DE"] = "MV"
    stepping_mode: Literal["candidate_eval", "online_rounds"] = "candidate_eval"

    # Control/drift model (Google-like approximate mapping).
    # `global`: one shared miscalibration scalar controls all gates.
    # `gate_specific`: gate instructions map to dedicated control slots.
    control_mode: Literal["global", "gate_specific"] = "gate_specific"
    control_dim: int = 8
    n_1q_control_slots: int = 24
    n_2q_control_slots: int = 24
    drift_period_steps: float = 150.0
    drift_amplitude: float = 1.0
    p_1q_base: float = 6.7e-4
    p_2q_base: float = 2.7e-3
    sensitivity_1q: float = 2.0e-3
    sensitivity_2q: float = 5.0e-3
    p_clip_max: float = 0.3
    # Parallel Monte-Carlo shots inside one simulator call.
    # 1 means no parallelism.
    shot_workers: int = 1
    # Performance knob:
    #   False (default): use lightweight summary run (fast, no per-shot traces).
    #   True: use full trace run for detector/stabilizer diagnostics (slow).
    collect_traces: bool = False
    reset_drift_on_episode: bool = False
    # Keep oracle-only metrics hidden by default to avoid leakage into RL policy.
    # Set True only for debugging/ablation studies.
    expose_oracle_metrics: bool = False
    seed: int = 42


def clipped_identity_action_mapper(theta: np.ndarray, action_limit: float = 2.0) -> np.ndarray:
    """Simple action mapper that only clips action range."""
    theta = np.asarray(theta, dtype=np.float32)
    return np.clip(theta, -float(action_limit), float(action_limit)).astype(np.float32)


class SteaneOnlineSteeringSimulator:
    """Protocol-compatible simulator for rl_train.ExternalSimulatorEnv.

    One `step(action)` evaluates one policy candidate with shot-averaging.
    The amount of simulated QEC per step depends on `cfg.stepping_mode`.
    """

    # obs = [success_rate, prep_ok_rate, mean(S1..S6), sin(phi), cos(phi), progress]
    # NOTE:
    #   We intentionally do NOT expose oracle metrics such as mse_to_opt in
    #   the default observation, because real hardware does not provide them.
    OBS_DIM = 11

    def __init__(self, cfg: SteaneAdapterConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._phase_offsets = self.rng.uniform(0.0, 2.0 * np.pi, size=self.action_dim)
        self._t = 0
        # Counts RL steps inside current episode (for online round mode).
        self._episode_step = 0

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    @property
    def action_dim(self) -> int:
        if self.cfg.control_mode == "global":
            return int(self.cfg.control_dim)
        if self.cfg.control_mode == "gate_specific":
            return int(self.cfg.n_1q_control_slots + self.cfg.n_2q_control_slots)
        raise ValueError(f"Unknown control_mode: {self.cfg.control_mode}")

    def _phase(self, t_step: int) -> float:
        if self.cfg.drift_period_steps <= 0:
            return 0.0
        return 2.0 * np.pi * float(t_step) / float(self.cfg.drift_period_steps)

    def _optimal_control(self, t_step: int) -> np.ndarray:
        phi = self._phase(t_step)
        return (self.cfg.drift_amplitude * np.sin(phi + self._phase_offsets)).astype(float)

    def _stabilizer_means_prep_ok(self, traces: Sequence[Dict], n_rounds_eval: int) -> np.ndarray:
        prep_ok_traces = [tr for tr in traces if bool(tr.get("prep_ok", False))]
        if not prep_ok_traces:
            return np.zeros(6, dtype=np.float32)

        out = np.zeros(6, dtype=float)
        denom = float(int(n_rounds_eval) * len(prep_ok_traces))
        for i, stab in enumerate(STABILIZER_SEQUENCE):
            name = stab["name"]
            ones = 0
            for tr in prep_ok_traces:
                ones += int(np.sum(tr.get("histories", {}).get(name, [])))
            out[i] = ones / denom if denom > 0 else 0.0
        return out.astype(np.float32)

    def _build_observation(
        self,
        success_rate: float,
        prep_ok_rate: float,
        stabilizer_means: np.ndarray,
        t_step: int,
        progress: float,
    ) -> np.ndarray:
        phi = self._phase(t_step)
        obs = np.array(
            [
                success_rate,
                prep_ok_rate,
                float(stabilizer_means[0]),
                float(stabilizer_means[1]),
                float(stabilizer_means[2]),
                float(stabilizer_means[3]),
                float(stabilizer_means[4]),
                float(stabilizer_means[5]),
                float(np.sin(phi)),
                float(np.cos(phi)),
                progress,
            ],
            dtype=np.float32,
        )
        return obs

    def _rounds_this_step(self) -> int:
        """Number of QEC rounds simulated in one RL step."""
        if self.cfg.stepping_mode == "candidate_eval":
            return int(self.cfg.n_rounds)
        if self.cfg.stepping_mode == "online_rounds":
            return 1
        raise ValueError(f"Unknown stepping_mode: {self.cfg.stepping_mode}")

    def _progress(self) -> float:
        """Episode progress in [0,1] for online-round mode; 0 otherwise."""
        if self.cfg.stepping_mode != "online_rounds":
            return 0.0
        return float(min(1.0, self._episode_step / max(1, int(self.cfg.n_rounds))))

    def reset(self) -> np.ndarray:
        """Reset simulator to a new RL episode."""
        # Keep global drift time across episodes by default (non-stationary setting).
        if self.cfg.reset_drift_on_episode:
            self._t = 0
        self._episode_step = 0
        obs = self._build_observation(
            success_rate=0.0,
            prep_ok_rate=0.0,
            stabilizer_means=np.zeros(6, dtype=np.float32),
            t_step=self._t,
            progress=self._progress(),
        )
        return obs.copy()

    def step(self, action: np.ndarray) -> SimulatorTransition:
        """Evaluate one policy candidate under current drift conditions."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[0]}")

        optimal = self._optimal_control(self._t)
        optimal_fn = lambda _t_ns: optimal
        if self.cfg.control_mode == "global":
            noise = GoogleLikeDepolarizingNoiseModel(
                control=action,
                optimal_control_fn=optimal_fn,
                p_1q_base=self.cfg.p_1q_base,
                p_2q_base=self.cfg.p_2q_base,
                sensitivity_1q=self.cfg.sensitivity_1q,
                sensitivity_2q=self.cfg.sensitivity_2q,
                p_clip_max=self.cfg.p_clip_max,
                enabled=True,
            )
        elif self.cfg.control_mode == "gate_specific":
            noise = GoogleLikeGateSpecificNoiseModel(
                control=action,
                optimal_control_fn=optimal_fn,
                p_1q_base=self.cfg.p_1q_base,
                p_2q_base=self.cfg.p_2q_base,
                sensitivity_1q=self.cfg.sensitivity_1q,
                sensitivity_2q=self.cfg.sensitivity_2q,
                n_1q_slots=self.cfg.n_1q_control_slots,
                n_2q_slots=self.cfg.n_2q_control_slots,
                p_clip_max=self.cfg.p_clip_max,
                enabled=True,
            )
        else:
            raise ValueError(f"Unknown control_mode: {self.cfg.control_mode}")
        p_1q, p_2q = noise.effective_error_rates(0.0)

        sim = SteaneQECSimulator(noise=noise)
        n_rounds_eval = self._rounds_this_step()
        n_steps = int(n_rounds_eval) * 6
        if self.cfg.collect_traces:
            # Full trace path: high-fidelity diagnostics, significantly slower.
            out = sim.run_experiment_with_trace(
                initial_state=self.cfg.initial_state,
                meas_basis=self.cfg.meas_basis,
                n_steps=n_steps,
                shots=int(self.cfg.shots_per_step),
                syndrome_mode=self.cfg.syndrome_mode,
                shot_workers=int(self.cfg.shot_workers),
            )
            traces = out["traces"]
            success_rate = float(out["success_rate"])
            prep_ok_count = int(np.sum([int(tr.get("prep_ok", False)) for tr in traces]))
            prep_ok_rate = prep_ok_count / float(self.cfg.shots_per_step)
            # Means are normalized by rounds actually simulated in this step.
            stabilizer_means = self._stabilizer_means_prep_ok(traces, n_rounds_eval=n_rounds_eval)
            detector_rates_for_reward = stabilizer_means.astype(float).tolist()
            metrics_source = "trace"
        else:
            # Fast path: summary-only run without storing shot traces.
            out = sim.run_experiment(
                initial_state=self.cfg.initial_state,
                meas_basis=self.cfg.meas_basis,
                n_steps=n_steps,
                shots=int(self.cfg.shots_per_step),
                syndrome_mode=self.cfg.syndrome_mode,
                shot_workers=int(self.cfg.shot_workers),
            )
            success_rate = float(out["success_rate"])
            # Not observable without traces; keep explicit sentinel value.
            prep_ok_rate = -1.0
            stabilizer_means = np.zeros(6, dtype=np.float32)
            # Use 1-success as coarse detector proxy to preserve reward signal.
            detector_rates_for_reward = [1.0 - success_rate]
            metrics_source = "success_proxy"
        miscalibration_mse = float(np.mean(np.square(action.astype(float) - optimal)))

        self._episode_step += 1
        progress = self._progress()
        done = bool(self.cfg.stepping_mode == "online_rounds" and self._episode_step >= int(self.cfg.n_rounds))

        obs = self._build_observation(
            success_rate=success_rate,
            prep_ok_rate=prep_ok_rate,
            stabilizer_means=stabilizer_means,
            t_step=self._t,
            progress=progress,
        )
        info: Dict[str, float | int | list[float]] = {
            "success_rate": success_rate,
            "prep_ok_rate": prep_ok_rate,
            "stabilizer_means": stabilizer_means.astype(float).tolist(),
            # Alias used by paper-style surrogate reward paths.
            "detector_rates": detector_rates_for_reward,
            "detector_rate_mean": float(np.mean(detector_rates_for_reward)),
            "p_1q": float(p_1q),
            "p_2q": float(p_2q),
            "time_index": int(self._t),
            "n_rounds": int(self.cfg.n_rounds),
            "n_steps": int(n_steps),
            "n_rounds_eval": int(n_rounds_eval),
            "shots": int(self.cfg.shots_per_step),
            "shot_workers": int(self.cfg.shot_workers),
            "collect_traces": bool(self.cfg.collect_traces),
            "metrics_source": metrics_source,
            "action_norm_l2": float(np.linalg.norm(action)),
            "control_mode": self.cfg.control_mode,
            "n_1q_control_slots": int(self.cfg.n_1q_control_slots),
            "n_2q_control_slots": int(self.cfg.n_2q_control_slots),
            "stepping_mode": self.cfg.stepping_mode,
            "episode_step": int(self._episode_step),
            "progress": progress,
        }
        # Oracle-only diagnostics are optionally exposed for offline debugging.
        if self.cfg.expose_oracle_metrics:
            info["miscalibration_mse"] = miscalibration_mse
        self._t += 1
        return SimulatorTransition(next_obs=obs, info=info, done=done)
