#!/usr/bin/env python3
"""Standalone Steane [[7,1,3]] QEC simulator.

This script mirrors the tutorial workflow but uses a sequential syndrome
measurement schedule:
    S1 -> S2 -> ... -> S6 -> S1 -> ...

Key features:
    - Logical state preparation from |0>_L into {+/-Z, +/-X, +/-Y}
    - n syndrome-measurement steps with fixed S1..S6 order
    - Per-stabilizer repeated measurements
    - Two aggregation modes for repeated checks: MV (majority vote) or DE
      (detection events from consecutive differences)
    - Pauli-frame updates via tutorial-style LUT decoder
    - Final destructive logical measurement in X/Y/Z basis
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import stim


LOGGER = logging.getLogger(__name__)

# Steane plaquettes in 0-based physical indexing.
PLAQUETTES = [
    [0, 1, 2, 3],  # first plaquette
    [1, 2, 4, 5],  # second plaquette
    [2, 3, 5, 6],  # third plaquette
]

# We keep the same X/Z ordering convention as the tutorial:
# S1,S2,S3 are X-type; S4,S5,S6 are Z-type.
STABILIZER_SEQUENCE = [
    {"name": "S1", "type": "X", "index": 0, "data": PLAQUETTES[0]},
    {"name": "S2", "type": "X", "index": 1, "data": PLAQUETTES[1]},
    {"name": "S3", "type": "X", "index": 2, "data": PLAQUETTES[2]},
    {"name": "S4", "type": "Z", "index": 0, "data": PLAQUETTES[0]},
    {"name": "S5", "type": "Z", "index": 1, "data": PLAQUETTES[1]},
    {"name": "S6", "type": "Z", "index": 2, "data": PLAQUETTES[2]},
]


@dataclass
class NoiseModel:
    """Pass-through placeholder noise model.

    Set enabled=True and extend `apply` when adding non-trivial noise.
    """

    enabled: bool = False

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        return circuit


def encoding_circuit(log_qb_idx: int = 0) -> stim.Circuit:
    """Goto-style Steane logical |0>_L preparation with ancilla verification."""
    c = stim.Circuit()
    s = log_qb_idx * 8

    c.append("H", [s, 4 + s, 6 + s])
    c.append("CNOT", [s, 1 + s])
    c.append("CNOT", [4 + s, 5 + s])
    c.append("CNOT", [6 + s, 3 + s])
    c.append("CNOT", [6 + s, 5 + s])
    c.append("CNOT", [4 + s, 2 + s])
    c.append("CNOT", [0 + s, 3 + s])
    c.append("CNOT", [4 + s, 1 + s])
    c.append("CNOT", [3 + s, 2 + s])

    # Verify logical Z_L via the dedicated ancilla (index 7+s).
    c.append("CNOT", [1 + s, 7 + s])
    c.append("CNOT", [3 + s, 7 + s])
    c.append("CNOT", [5 + s, 7 + s])
    c.append("M", [7 + s])
    return c


def logical_single_qubit_gate(gate: str, log_qb_idx: int = 0) -> stim.Circuit:
    """Apply a transversal logical single-qubit Clifford gate on Steane block."""
    c = stim.Circuit()
    s = log_qb_idx * 8

    if gate == "Z":
        c.append("Z", np.array([4, 5, 6]) + s)
    elif gate == "X":
        c.append("X", np.array([4, 5, 6]) + s)
    elif gate == "H":
        c.append("H", np.array([0, 1, 2, 3, 4, 5, 6]) + s)
    elif gate == "S":
        c.append("S", np.array([0, 1, 2, 3, 4, 5, 6]) + s)
    elif gate == "S_DAG":
        c.append("S_DAG", np.array([0, 1, 2, 3, 4, 5, 6]) + s)
    else:
        raise ValueError(f"Unknown logical gate: {gate}")

    return c


def prepare_stab_eigenstate(stabilizer: str) -> stim.Circuit:
    """Prepare desired logical eigenstate from encoded |0>_L."""
    c = stim.Circuit()

    if stabilizer == "+Z":  # |0>_L
        pass
    elif stabilizer == "-Z":  # |1>_L
        c += logical_single_qubit_gate(gate="X")
    elif stabilizer == "+X":  # |+>_L
        c += logical_single_qubit_gate(gate="H")
    elif stabilizer == "-X":  # |->_L
        c += logical_single_qubit_gate(gate="X")
        c += logical_single_qubit_gate(gate="H")
    elif stabilizer == "+Y":  # |+i>_L
        c += logical_single_qubit_gate(gate="H")
        c += logical_single_qubit_gate(gate="S")
    elif stabilizer == "-Y":  # |-i>_L
        c += logical_single_qubit_gate(gate="X")
        c += logical_single_qubit_gate(gate="H")
        c += logical_single_qubit_gate(gate="S")
    else:
        raise ValueError(f"Unknown stabilizer/eigenstate label: {stabilizer}")

    return c


def rotate_to_measurement_basis(meas_basis: str) -> stim.Circuit:
    """Rotate logical qubit so final physical Z-measurement yields target basis."""
    c = stim.Circuit()
    if meas_basis == "Z":
        pass
    elif meas_basis == "X":
        c += logical_single_qubit_gate(gate="H")
    elif meas_basis == "Y":
        c += logical_single_qubit_gate(gate="S_DAG")
        c += logical_single_qubit_gate(gate="H")
    else:
        raise ValueError(f"Unknown measurement basis: {meas_basis}")
    return c


def measure_logical_qubits(log_qubit_indices: Optional[list] = None) -> stim.Circuit:
    """Destructively measure selected logical blocks in physical Z basis."""
    if log_qubit_indices is None:
        log_qubit_indices = [0]

    c = stim.Circuit()
    for log_qubit_index in log_qubit_indices:
        s = log_qubit_index * 8
        c.append("M", np.array([0, 1, 2, 3, 4, 5, 6]) + s)
    return c


def measure_single_stabilizer(stab: dict, ancilla: int = 8) -> stim.Circuit:
    """Measure one stabilizer using one ancilla, then reset ancilla."""
    c = stim.Circuit()
    data = stab["data"]
    stab_type = stab["type"]

    if stab_type == "X":
        # Tutorial convention for X-type round.
        for q in data:
            c.append("CNOT", [q, ancilla])
    elif stab_type == "Z":
        # Tutorial convention for Z-type round.
        c.append("H", ancilla)
        for q in data:
            c.append("CNOT", [ancilla, q])
        c.append("H", ancilla)
    else:
        raise ValueError(f"Invalid stabilizer type: {stab_type}")

    c.append("M", ancilla)
    c.append("R", ancilla)
    return c


def _majority_vote(bits: list[int]) -> int:
    if not bits:
        return 0
    ones = int(np.sum(bits))
    zeros = len(bits) - ones
    return int(ones > zeros)


def _detect_event(bits: list[int]) -> int:
    if len(bits) < 2:
        return 0
    return int(bits[-1] ^ bits[-2])


def aggregate_syndromes(
    histories: dict[str, list[int]], mode: Literal["MV", "DE"]
) -> tuple[np.ndarray, np.ndarray]:
    """Build effective X/Z syndrome vectors from repeated measurements."""
    x = np.zeros(3, dtype=int)
    z = np.zeros(3, dtype=int)

    for stab in STABILIZER_SEQUENCE:
        bits = histories[stab["name"]]
        if mode == "MV":
            val = _majority_vote(bits)
        elif mode == "DE":
            val = _detect_event(bits)
        else:
            raise ValueError(f"Unknown syndrome mode: {mode}")

        if stab["type"] == "X":
            x[stab["index"]] = val
        else:
            z[stab["index"]] = val

    return x, z


def unflagged_decoder(syndromes: np.ndarray) -> int:
    """Tutorial LUT decoder for one error type (X or Z channel separately)."""
    bad_syndrome_patterns = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1]])
    if np.any(np.all(bad_syndrome_patterns == syndromes, axis=1)):
        LOGGER.debug("Decoder: logical error detected from syndrome=%s", syndromes)
        return 1
    return 0


def expected_result(measure_output: int, initial_state: str, meas_basis: str) -> int:
    """Check whether measurement matches expected eigenvalue relation."""
    pauli_measurement = stim.PauliString(meas_basis)
    pauli_stabilizer = stim.PauliString(initial_state.replace("+", "").replace("-", ""))
    commute = pauli_stabilizer.commutes(pauli_measurement)

    if not commute:
        raise ValueError(
            f"Initial state {initial_state} and basis {meas_basis} anti-commute; "
            "measurement should be random."
        )

    sign = 1 if initial_state.startswith("+") else -1
    expected = 0 if sign == 1 else 1
    return int(measure_output == expected)


def destructive_logical_measurement(
    simulator: stim.TableauSimulator,
    meas_basis: str,
    tracked_x_syndromes: np.ndarray,
    tracked_z_syndromes: np.ndarray,
    pauli_frame: np.ndarray,
    m_idx: int,
    noise: NoiseModel,
) -> tuple[int, int]:
    """Final logical readout plus final decoder update from physical readout."""
    simulator.do(noise.apply(measure_logical_qubits()))
    r = simulator.current_measurement_record()[m_idx : m_idx + 7]
    m_idx += 7

    # Logical Z_L observable from data qubits 4,5,6 (tutorial convention).
    log_obs = int(r[4] ^ r[5] ^ r[6])

    # Reconstruct three plaquette parities from final destructive measurements.
    s1 = int(r[0] ^ r[1] ^ r[2] ^ r[3])
    s2 = int(r[1] ^ r[2] ^ r[4] ^ r[5])
    s3 = int(r[2] ^ r[3] ^ r[5] ^ r[6])
    syndromes = np.array([s1, s2, s3], dtype=int)

    if meas_basis == "X":
        syndrome_diff = syndromes ^ tracked_x_syndromes
    elif meas_basis == "Y":
        syndrome_diff = syndromes ^ tracked_x_syndromes ^ tracked_z_syndromes
    elif meas_basis == "Z":
        syndrome_diff = syndromes ^ tracked_z_syndromes
    else:
        raise ValueError(f"Unknown measurement basis: {meas_basis}")

    final_correction = unflagged_decoder(syndrome_diff)
    log_obs ^= final_correction

    # Apply Pauli-frame correction in the requested logical basis.
    if meas_basis == "X":
        log_obs ^= int(pauli_frame[0])
    elif meas_basis == "Y":
        log_obs ^= int(pauli_frame[0] ^ pauli_frame[1])
    elif meas_basis == "Z":
        log_obs ^= int(pauli_frame[1])

    return log_obs, m_idx


def steane_code_exp_sequential(
    initial_state: str = "+Z",
    meas_basis: str = "Z",
    n_steps: int = 60,
    shots: int = 100,
    syndrome_mode: Literal["MV", "DE"] = "MV",
    noise: Optional[NoiseModel] = None,
) -> list[int]:
    """Run end-to-end Steane QEC experiment with sequential S1..S6 schedule."""
    if noise is None:
        noise = NoiseModel(enabled=False)

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    results: list[int] = []

    for _ in range(shots):
        simulator = stim.TableauSimulator()
        m_idx = 0

        # Up to 3 preparation attempts, matching tutorial style.
        prep_ok = False
        for _attempt in range(3):
            simulator.do(noise.apply(encoding_circuit()))
            state_prep_ancilla = int(simulator.current_measurement_record()[m_idx])
            m_idx += 1
            if state_prep_ancilla == 0:
                prep_ok = True
                break
        if not prep_ok:
            results.append(0)
            continue

        simulator.do(noise.apply(prepare_stab_eigenstate(initial_state)))

        histories: dict[str, list[int]] = {s["name"]: [] for s in STABILIZER_SEQUENCE}
        tracked_x_syndromes = np.zeros(3, dtype=int)
        tracked_z_syndromes = np.zeros(3, dtype=int)
        # Pauli frame stores pending logical corrections [X_L, Z_L].
        pauli_frame = np.array([0, 0], dtype=int)

        for step in range(n_steps):
            stab = STABILIZER_SEQUENCE[step % 6]
            simulator.do(noise.apply(measure_single_stabilizer(stab, ancilla=8)))
            meas = int(simulator.current_measurement_record()[m_idx])
            m_idx += 1
            histories[stab["name"]].append(meas)

            # Update decoder every full S1..S6 pass.
            if step % 6 == 5:
                current_x, current_z = aggregate_syndromes(histories, syndrome_mode)
                diff_x = tracked_x_syndromes ^ current_x
                diff_z = tracked_z_syndromes ^ current_z
                pauli_frame[0] ^= unflagged_decoder(diff_x)
                pauli_frame[1] ^= unflagged_decoder(diff_z)
                tracked_x_syndromes = current_x
                tracked_z_syndromes = current_z

        # If there is an unfinished partial S1..S6 block, fold it in once.
        if n_steps % 6 != 0:
            current_x, current_z = aggregate_syndromes(histories, syndrome_mode)
            diff_x = tracked_x_syndromes ^ current_x
            diff_z = tracked_z_syndromes ^ current_z
            pauli_frame[0] ^= unflagged_decoder(diff_x)
            pauli_frame[1] ^= unflagged_decoder(diff_z)
            tracked_x_syndromes = current_x
            tracked_z_syndromes = current_z

        simulator.do(noise.apply(rotate_to_measurement_basis(meas_basis)))
        final_measurement, m_idx = destructive_logical_measurement(
            simulator=simulator,
            meas_basis=meas_basis,
            tracked_x_syndromes=tracked_x_syndromes,
            tracked_z_syndromes=tracked_z_syndromes,
            pauli_frame=pauli_frame,
            m_idx=m_idx,
            noise=noise,
        )

        success = expected_result(final_measurement, initial_state, meas_basis)
        results.append(success)

    return results


class SteaneQECSimulator:
    """High-level interface for Steane sequential-QEC experiments."""

    def __init__(self, noise: Optional[NoiseModel] = None):
        self.noise = noise if noise is not None else NoiseModel(enabled=False)
        self._last_run: Optional[dict[str, Any]] = None

    # ---- Circuit builders (tutorial-style convenience wrappers) ----
    def build_encoding_circuit(self, log_qb_idx: int = 0) -> stim.Circuit:
        return encoding_circuit(log_qb_idx=log_qb_idx)

    def build_logical_gate(self, gate: str, log_qb_idx: int = 0) -> stim.Circuit:
        return logical_single_qubit_gate(gate=gate, log_qb_idx=log_qb_idx)

    def build_state_prep_circuit(self, initial_state: str = "+Z") -> stim.Circuit:
        c = stim.Circuit()
        c += encoding_circuit()
        c += prepare_stab_eigenstate(initial_state)
        return c

    def build_single_stabilizer_circuit(self, stabilizer: str, ancilla: int = 8) -> stim.Circuit:
        names = [s["name"] for s in STABILIZER_SEQUENCE]
        if stabilizer not in names:
            raise ValueError(f"Unknown stabilizer {stabilizer}. Choices: {names}")
        stab = next(s for s in STABILIZER_SEQUENCE if s["name"] == stabilizer)
        return measure_single_stabilizer(stab, ancilla=ancilla)

    def build_syndrome_schedule_circuit(self, n_steps: int = 60, ancilla: int = 8) -> stim.Circuit:
        c = stim.Circuit()
        for step in range(n_steps):
            stab = STABILIZER_SEQUENCE[step % 6]
            c += measure_single_stabilizer(stab, ancilla=ancilla)
        return c

    def build_full_flow_circuit_for_diagram(
        self,
        initial_state: str = "+Z",
        meas_basis: str = "Z",
        n_steps: int = 12,
    ) -> stim.Circuit:
        """Build a static circuit useful for visual inspection/diagram only."""
        c = stim.Circuit()
        c += encoding_circuit()
        c += prepare_stab_eigenstate(initial_state)
        c += self.build_syndrome_schedule_circuit(n_steps=n_steps, ancilla=8)
        c += rotate_to_measurement_basis(meas_basis)
        c += measure_logical_qubits()
        return c

    # ---- Diagram helpers ----
    def get_diagram(self, circuit: stim.Circuit, diagram_type: str = "timeline-text") -> str:
        return str(circuit.diagram(diagram_type))

    def print_diagram(self, circuit: stim.Circuit, diagram_type: str = "timeline-text") -> None:
        print(self.get_diagram(circuit, diagram_type=diagram_type))

    # ---- Validation helpers ----
    def validate_encoding(self, shots: int = 100) -> dict[str, Any]:
        """Sanity-check state-prep ancilla and logical Z_L readout."""
        c = stim.Circuit()
        c += encoding_circuit()
        c += measure_logical_qubits()
        sampler = c.compile_sampler()
        r = sampler.sample(shots=shots).astype(int)

        ancilla = r[:, 0]
        # Following tutorial indexing convention in this circuit:
        # columns 1..7 are data-qubit measurements.
        logical_z = r[:, 6] ^ r[:, 7] ^ r[:, 8] if r.shape[1] >= 9 else r[:, 5] ^ r[:, 6] ^ r[:, 7]
        # The expression above keeps compatibility if sampler shape differs by version.

        out = {
            "shots": shots,
            "ancilla_zero_rate": float(np.mean(ancilla == 0)),
            "logical_zero_rate": float(np.mean(logical_z == 0)),
        }
        return out

    def validate_prepared_state(
        self,
        initial_state: str = "+Z",
        meas_basis: str = "Z",
        shots: int = 100,
    ) -> dict[str, Any]:
        """Prepare a logical state and validate expected basis measurement."""
        c = stim.Circuit()
        c += encoding_circuit()
        c += prepare_stab_eigenstate(initial_state)
        c += rotate_to_measurement_basis(meas_basis)
        c += measure_logical_qubits()

        sampler = c.compile_sampler()
        r = sampler.sample(shots=shots).astype(int)
        logical_measurements = r[:, 5] ^ r[:, 6] ^ r[:, 7]
        successes = [
            expected_result(int(m), initial_state=initial_state, meas_basis=meas_basis)
            for m in logical_measurements
        ]
        return {
            "shots": shots,
            "initial_state": initial_state,
            "meas_basis": meas_basis,
            "success_rate": float(np.mean(successes)),
            "logical_measurements": logical_measurements.tolist(),
        }

    # ---- Experiment runner / results ----
    def run_experiment(
        self,
        initial_state: str = "+Z",
        meas_basis: str = "Z",
        n_steps: int = 60,
        shots: int = 100,
        syndrome_mode: Literal["MV", "DE"] = "MV",
    ) -> dict[str, Any]:
        results = steane_code_exp_sequential(
            initial_state=initial_state,
            meas_basis=meas_basis,
            n_steps=n_steps,
            shots=shots,
            syndrome_mode=syndrome_mode,
            noise=self.noise,
        )
        out = {
            "initial_state": initial_state,
            "meas_basis": meas_basis,
            "n_steps": n_steps,
            "shots": shots,
            "syndrome_mode": syndrome_mode,
            "results": results,
            "success_rate": float(np.mean(results)) if results else 0.0,
        }
        self._last_run = out
        return out

    def get_results(self) -> dict[str, Any]:
        if self._last_run is None:
            raise RuntimeError("No experiment has been run yet. Call run_experiment first.")
        return self._last_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Steane sequential-QEC simulator")
    parser.add_argument("--initial-state", default="+Z", choices=["+Z", "-Z", "+X", "-X", "+Y", "-Y"])
    parser.add_argument("--meas-basis", default="Z", choices=["X", "Y", "Z"])
    parser.add_argument("--n-steps", type=int, default=60, help="Total syndrome-measurement steps.")
    parser.add_argument("--shots", type=int, default=200, help="Number of Monte-Carlo shots.")
    parser.add_argument(
        "--mode",
        default="MV",
        choices=["MV", "DE"],
        help="Syndrome aggregation mode: MV (majority vote) or DE (detection event).",
    )
    parser.add_argument(
        "--print-diagram",
        default="none",
        choices=["none", "encoding", "syndrome", "full"],
        help="Print selected circuit diagram and exit.",
    )
    parser.add_argument(
        "--diagram-type",
        default="timeline-text",
        help="Stim diagram type (e.g. timeline-text, timeline-svg, timeline-svg-html).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run quick tutorial-style validation checks before experiment.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(message)s",
    )

    sim = SteaneQECSimulator(noise=NoiseModel(enabled=False))

    if args.print_diagram != "none":
        if args.print_diagram == "encoding":
            circuit = sim.build_encoding_circuit()
        elif args.print_diagram == "syndrome":
            circuit = sim.build_syndrome_schedule_circuit(n_steps=args.n_steps)
        else:
            circuit = sim.build_full_flow_circuit_for_diagram(
                initial_state=args.initial_state,
                meas_basis=args.meas_basis,
                n_steps=args.n_steps,
            )
        sim.print_diagram(circuit, diagram_type=args.diagram_type)
        return

    if args.validate:
        enc_validation = sim.validate_encoding(shots=min(500, args.shots))
        state_validation = sim.validate_prepared_state(
            initial_state=args.initial_state,
            meas_basis=args.meas_basis,
            shots=min(500, args.shots),
        )
        print("Validation summary")
        print(f"  ancilla_zero_rate: {enc_validation['ancilla_zero_rate']:.4f}")
        print(f"  logical_zero_rate: {enc_validation['logical_zero_rate']:.4f}")
        print(f"  prepared_state_success_rate: {state_validation['success_rate']:.4f}")

    summary = sim.run_experiment(
        initial_state=args.initial_state,
        meas_basis=args.meas_basis,
        n_steps=args.n_steps,
        shots=args.shots,
        syndrome_mode=args.mode,
    )

    print("Steane sequential-QEC summary")
    print(f"  initial_state: {args.initial_state}")
    print(f"  meas_basis:    {args.meas_basis}")
    print(f"  n_steps:       {args.n_steps}")
    print(f"  shots:         {args.shots}")
    print(f"  mode:          {args.mode}")
    print(f"  success_rate:  {summary['success_rate']:.4f}")


if __name__ == "__main__":
    main()
