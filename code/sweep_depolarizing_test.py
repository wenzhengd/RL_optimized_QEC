#!/usr/bin/env python3
"""Sweep time-independent depolarizing noise and export PDF plots.

Requested default experiment:
  - n_rounds = 10
  - n_shots = 100
  - error_rate array = [0.00, 0.05, ..., 0.50]

For each error budget p_total, we model an idle-layer depolarizing channel:
  E_k(rho) = (1 - p_idle) rho + (p_idle/3) X rho X + (p_idle/3) Y rho Y + (p_idle/3) Z rho Z
where p_idle = p_total / N_idle(successful_shot)

The project noise engine expects per-ns rates integrated over the idle window.
So for idle length T_idle, each axis rate is set to:
  rate_x = rate_y = rate_z = (p_idle / 3) / T_idle
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from noise_engine import GateDurations, TimeDependentPauliNoiseModel
from steane_code_simulator import (
    STABILIZER_SEQUENCE,
    SteaneQECSimulator,
    encoding_circuit,
    measure_logical_qubits,
    measure_single_stabilizer,
    prepare_stab_eigenstate,
    rotate_to_measurement_basis,
)


class CachedTimeDependentPauliNoiseModel(TimeDependentPauliNoiseModel):
    """TimeDependentPauliNoiseModel with circuit-level apply cache."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._cache: dict[str, Any] = {}

    def apply(self, circuit: Any) -> Any:
        if not self.enabled:
            return circuit
        key = str(circuit)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        out = super().apply(circuit)
        self._cache[key] = out
        return out


DEFAULT_ERROR_RATE_ARRAY = np.round(np.arange(0.0, 0.5000001, 0.05), 2)


def parse_error_rate_array(text: str) -> np.ndarray:
    """Parse comma-separated total error budgets."""
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if not parts:
        raise ValueError("error-rates is empty after parsing")
    vals = np.array([float(x) for x in parts], dtype=float)
    if np.any(vals < 0.0) or np.any(vals > 1.0):
        raise ValueError("All error rates must satisfy 0 <= p <= 1")
    return vals


def _count_idle_windows(circuit: Any) -> int:
    """Count idle windows inserted by the noise engine for this circuit."""
    n_inst = sum(1 for _ in circuit)
    return max(n_inst - 1, 0)


def estimate_idle_windows_per_successful_shot(
    n_steps: int,
    initial_state: str,
    meas_basis: str,
) -> int:
    """Estimate idle windows for one successful shot execution path."""
    total_idle = 0
    total_idle += _count_idle_windows(encoding_circuit())
    total_idle += _count_idle_windows(prepare_stab_eigenstate(initial_state))

    stab_idle = {
        stab["name"]: _count_idle_windows(measure_single_stabilizer(stab, ancilla=8))
        for stab in STABILIZER_SEQUENCE
    }
    for step in range(n_steps):
        stab_name = STABILIZER_SEQUENCE[step % 6]["name"]
        total_idle += stab_idle[stab_name]

    total_idle += _count_idle_windows(rotate_to_measurement_basis(meas_basis))
    total_idle += _count_idle_windows(measure_logical_qubits())
    return total_idle


def compute_stabilizer_averages(
    traces: list[dict[str, Any]],
    stabilizer_names: list[str],
    n_rounds: int,
) -> dict[str, float]:
    """Average stabilizer measurement values over rounds and shots.

    This is conditioned on prep_ok shots only, so prep-failed traces do not
    artificially pull the averages downward.
    """
    prep_ok_traces = [t for t in traces if bool(t.get("prep_ok", False))]
    denom = float(n_rounds * len(prep_ok_traces))
    out: dict[str, float] = {}
    for name in stabilizer_names:
        total_ones = 0
        for trace in prep_ok_traces:
            hist = trace.get("histories", {}).get(name, [])
            total_ones += int(np.sum(hist))
        out[name] = total_ones / denom if denom > 0 else 0.0
    return out


def run_sweep(
    n_rounds: int,
    n_shots: int,
    error_rates: np.ndarray,
    initial_state: str,
    meas_basis: str,
    syndrome_mode: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Run sweep and return stabilizer means + success rates."""
    n_stab = 6
    n_steps = n_rounds * n_stab
    stabilizer_names = [s["name"] for s in STABILIZER_SEQUENCE]

    stab_means = np.zeros((len(stabilizer_names), len(error_rates)), dtype=float)
    success_rates = np.zeros(len(error_rates), dtype=float)

    idle_ns = GateDurations().idle_ns
    idle_windows_per_successful_shot = estimate_idle_windows_per_successful_shot(
        n_steps=n_steps,
        initial_state=initial_state,
        meas_basis=meas_basis,
    )
    print(
        f"Running sweep: n_rounds={n_rounds}, n_steps={n_steps}, n_shots={n_shots}, "
        f"n_points={len(error_rates)}, idle_windows_per_successful_shot={idle_windows_per_successful_shot}",
        flush=True,
    )

    for i, p_total in enumerate(error_rates):
        p_idle = p_total / float(idle_windows_per_successful_shot)
        rate_each_axis = (p_idle / 3.0) / idle_ns
        noise = CachedTimeDependentPauliNoiseModel(
            p_x=rate_each_axis,
            p_y=rate_each_axis,
            p_z=rate_each_axis,
            enabled=True,
        )
        sim = SteaneQECSimulator(noise=noise)
        out = sim.run_experiment_with_trace(
            initial_state=initial_state,
            meas_basis=meas_basis,
            n_steps=n_steps,
            shots=n_shots,
            syndrome_mode=syndrome_mode,  # type: ignore[arg-type]
        )

        success_rates[i] = float(out["success_rate"])
        means = compute_stabilizer_averages(
            traces=out["traces"],
            stabilizer_names=stabilizer_names,
            n_rounds=n_rounds,
        )
        for j, name in enumerate(stabilizer_names):
            stab_means[j, i] = means[name]

        prep_ok_count = int(np.sum([int(t.get("prep_ok", False)) for t in out["traces"]]))
        prep_ok_rate = prep_ok_count / float(n_shots)

        print(
            f"[{i + 1}/{len(error_rates)}] p_total={p_total:.2f} p_idle={p_idle:.6f} "
            f"prep_ok_rate={prep_ok_rate:.4f} success_rate={success_rates[i]:.4f}",
            flush=True,
        )

    return stabilizer_names, stab_means, success_rates


def _run_single_error_rate_point(args: tuple[Any, ...]) -> tuple[int, float, dict[str, float], float, float]:
    """Worker entry for one error-rate point."""
    (
        idx,
        p_total,
        n_rounds,
        n_shots,
        initial_state,
        meas_basis,
        syndrome_mode,
        idle_ns,
        idle_windows_per_successful_shot,
        stabilizer_names,
    ) = args

    # Avoid oversubscription when running many worker processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    p_idle = float(p_total) / float(idle_windows_per_successful_shot)
    rate_each_axis = (p_idle / 3.0) / float(idle_ns)
    noise = CachedTimeDependentPauliNoiseModel(
        p_x=rate_each_axis,
        p_y=rate_each_axis,
        p_z=rate_each_axis,
        enabled=True,
    )
    sim = SteaneQECSimulator(noise=noise)
    out = sim.run_experiment_with_trace(
        initial_state=initial_state,
        meas_basis=meas_basis,
        n_steps=int(n_rounds) * 6,
        shots=int(n_shots),
        syndrome_mode=syndrome_mode,  # type: ignore[arg-type]
    )

    success_rate = float(out["success_rate"])
    means = compute_stabilizer_averages(
        traces=out["traces"],
        stabilizer_names=list(stabilizer_names),
        n_rounds=int(n_rounds),
    )
    prep_ok_count = int(np.sum([int(t.get("prep_ok", False)) for t in out["traces"]]))
    prep_ok_rate = prep_ok_count / float(n_shots)
    return int(idx), float(p_total), means, prep_ok_rate, success_rate


def run_sweep_parallel(
    n_rounds: int,
    n_shots: int,
    error_rates: np.ndarray,
    initial_state: str,
    meas_basis: str,
    syndrome_mode: str,
    workers: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Run sweep in parallel over error-rate points."""
    n_stab = 6
    n_steps = n_rounds * n_stab
    stabilizer_names = [s["name"] for s in STABILIZER_SEQUENCE]

    stab_means = np.zeros((len(stabilizer_names), len(error_rates)), dtype=float)
    success_rates = np.zeros(len(error_rates), dtype=float)

    idle_ns = GateDurations().idle_ns
    idle_windows_per_successful_shot = estimate_idle_windows_per_successful_shot(
        n_steps=n_steps,
        initial_state=initial_state,
        meas_basis=meas_basis,
    )
    print(
        f"Running parallel sweep: n_rounds={n_rounds}, n_steps={n_steps}, n_shots={n_shots}, "
        f"n_points={len(error_rates)}, workers={workers}, "
        f"idle_windows_per_successful_shot={idle_windows_per_successful_shot}",
        flush=True,
    )

    task_args: list[tuple[Any, ...]] = []
    for i, p_total in enumerate(error_rates):
        task_args.append(
            (
                i,
                float(p_total),
                int(n_rounds),
                int(n_shots),
                initial_state,
                meas_basis,
                syndrome_mode,
                float(idle_ns),
                int(idle_windows_per_successful_shot),
                stabilizer_names,
            )
        )

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, p_total, means, prep_ok_rate, success_rate in ex.map(_run_single_error_rate_point, task_args):
            success_rates[idx] = success_rate
            for j, name in enumerate(stabilizer_names):
                stab_means[j, idx] = means[name]
            p_idle = p_total / float(idle_windows_per_successful_shot)
            print(
                f"[{idx + 1}/{len(error_rates)}] p_total={p_total:.2f} p_idle={p_idle:.6f} "
                f"prep_ok_rate={prep_ok_rate:.4f} success_rate={success_rate:.4f}",
                flush=True,
            )

    return stabilizer_names, stab_means, success_rates


def save_stabilizer_plot(
    output_path: Path,
    error_rates: np.ndarray,
    stabilizer_names: list[str],
    stab_means: np.ndarray,
) -> None:
    """Save stabilizer average syndrome plot as PDF."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for i, name in enumerate(stabilizer_names):
        plt.plot(error_rates, stab_means[i], marker="o", linewidth=1.6, label=name)
    plt.xlabel("Total error budget p_total per successful shot")
    plt.ylabel("Average syndrome measurement value")
    plt.title("Steane Stabilizer Averages (prep_ok shots) vs Total Error Budget")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.close()


def save_success_plot(output_path: Path, error_rates: np.ndarray, success_rates: np.ndarray) -> None:
    """Save final success rate plot as PDF."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(error_rates, success_rates, marker="o", linewidth=1.8, color="tab:blue")
    plt.xlabel("Total error budget p_total per successful shot")
    plt.ylabel("Final success rate")
    plt.title("Steane Final Success Rate vs Total Error Budget")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steane depolarizing-noise sweep (PDF outputs).")
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--n-shots", type=int, default=100)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers over error-rate points. 0 means auto.",
    )
    parser.add_argument(
        "--error-rates",
        default="",
        help="Comma-separated total error budgets. Default uses built-in array.",
    )
    parser.add_argument("--initial-state", default="+Z", choices=["+Z", "-Z", "+X", "-X", "+Y", "-Y"])
    parser.add_argument("--meas-basis", default="Z", choices=["X", "Y", "Z"])
    parser.add_argument("--mode", default="MV", choices=["MV", "DE"])
    parser.add_argument("--out-dir", default=".")
    parser.add_argument(
        "--stabilizer-pdf",
        default="stabilizer_avg_syndrome_vs_error_rate.pdf",
        help="Filename for the stabilizer averages PDF.",
    )
    parser.add_argument(
        "--success-pdf",
        default="success_rate_vs_error_rate.pdf",
        help="Filename for the success-rate PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_rounds < 1 or args.n_shots < 1:
        raise ValueError("n_rounds and n_shots must be >= 1")

    if args.error_rates.strip():
        error_rates = parse_error_rate_array(args.error_rates)
    else:
        error_rates = DEFAULT_ERROR_RATE_ARRAY.copy()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.workers < 0:
        raise ValueError("workers must be >= 0")

    if args.workers == 0:
        cpu = os.cpu_count() or 1
        workers = max(1, min(len(error_rates), cpu - 1 if cpu > 1 else 1))
    else:
        workers = min(args.workers, len(error_rates))

    if workers <= 1:
        stabilizer_names, stab_means, success_rates = run_sweep(
            n_rounds=args.n_rounds,
            n_shots=args.n_shots,
            error_rates=error_rates,
            initial_state=args.initial_state,
            meas_basis=args.meas_basis,
            syndrome_mode=args.mode,
        )
    else:
        stabilizer_names, stab_means, success_rates = run_sweep_parallel(
            n_rounds=args.n_rounds,
            n_shots=args.n_shots,
            error_rates=error_rates,
            initial_state=args.initial_state,
            meas_basis=args.meas_basis,
            syndrome_mode=args.mode,
            workers=workers,
        )

    stab_pdf_path = out_dir / args.stabilizer_pdf
    success_pdf_path = out_dir / args.success_pdf

    save_stabilizer_plot(
        output_path=stab_pdf_path,
        error_rates=error_rates,
        stabilizer_names=stabilizer_names,
        stab_means=stab_means,
    )
    save_success_plot(
        output_path=success_pdf_path,
        error_rates=error_rates,
        success_rates=success_rates,
    )

    print(f"Saved: {stab_pdf_path}")
    print(f"Saved: {success_pdf_path}")


if __name__ == "__main__":
    main()
