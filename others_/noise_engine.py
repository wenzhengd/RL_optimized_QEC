#!/usr/bin/env python3
"""Noise engine for circuit-level time-dependent Pauli channels.

This module is intentionally independent from any specific code (e.g. Steane),
so collaborators can reuse the same noise model across different circuits.

Well-defined modeling choices implemented here
----------------------------------------------
1) Serial schedule:
   Instructions are executed one-by-one in textual order (no parallel layers).

2) Gate durations (ns):
   - 1-qubit gate: 10
   - 2-qubit gate: 20
   - measurement: 100
   - reset: 100
   These are configurable via `GateDurations`.

3) Idle duration:
   A fixed idle window is inserted between consecutive instructions:
       [end(op_k), start(op_{k+1})] with length `idle_ns` (default 200 ns).
   No idle window is added before the first operation or after the last.

4) Noise placement:
   No noise is injected during operations.
   Noise is injected only during idle windows.

5) Channel form:
   For each physical qubit j and idle time t, users provide time-dependent
   Pauli coefficients/rates pX_j(t), pY_j(t), pZ_j(t), which may be unequal.
   Over one idle interval [t0, t1], this module computes:
       P_r(j) = integral_{t0}^{t1} p_r_j(t) dt
   then appends PAULI_CHANNEL_1 with [P_X, P_Y, P_Z] on each qubit.

6) Physical validity:
   For each qubit and idle window, probabilities are validated:
       P_X >= 0, P_Y >= 0, P_Z >= 0, P_X + P_Y + P_Z <= 1
   (within tolerance). Violations raise ValueError by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import stim

# Type: f(t_ns, qubit_index) -> instantaneous coefficient/rate.
RateFn = Callable[[float, int], float]
RateSpec = Union[float, RateFn]


@dataclass
class GateDurations:
    """Durations (in ns) used by the serial timeline builder."""

    one_qubit_ns: float = 10.0 #ns
    two_qubit_ns: float = 20.0 #ns
    measure_ns: float = 100.0  #ns
    reset_ns: float = 100.0    #ns
    idle_ns: float = 200.0     #ns


@dataclass
class OperationEvent:
    """A single operation projected onto the serial timeline."""

    instruction: stim.CircuitInstruction
    start_ns: float
    end_ns: float


@dataclass
class NoiseModel:
    """
    Noiseless/pass-through base class.

    This keeps the existing `noise.apply(circuit)` interface stable.
    """

    enabled: bool = False

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """Return circuit unchanged (noiseless baseline)."""
        return circuit


def compile_time_expression(expr: str) -> RateFn:
    """Compile a user expression into a callable rate function.

    The expression can use variables:
      - `t`: time in ns
      - `q`: qubit index

    Allowed math symbols include a safe subset from numpy, e.g.
    `sin`, `cos`, `exp`, `log`, `sqrt`, `pi`.

    Example:
        f = compile_time_expression("1e-6 * (1 + 0.2*sin(2*pi*t/1000))")
        value = f(200.0, 3)

    Notes:
      - This helper is for trusted expressions.
      - If you need strict sandboxing for expressions, replace with a parser.
    """

    allowed = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
    }

    def _fn(t_ns: float, qubit_index: int) -> float:
        local_scope = dict(allowed)
        local_scope["t"] = float(t_ns)
        local_scope["q"] = int(qubit_index)
        return float(eval(expr, {"__builtins__": {}}, local_scope))

    return _fn


def _as_rate_fn(spec: RateSpec) -> RateFn:
    """Normalize scalar/callable rate spec to a callable."""
    if callable(spec):
        return spec

    constant = float(spec)

    def _const_rate(_t_ns: float, _q: int) -> float:
        return constant

    return _const_rate


def _iter_flat_instructions(circuit: stim.Circuit) -> List[stim.CircuitInstruction]:
    """Expand a stim circuit into a flat list of instructions.

    Repeat blocks are recursively unrolled. This matches the serial scheduling
    assumption and simplifies timeline/noise insertion.
    """

    flat: List[stim.CircuitInstruction] = []
    for op in circuit:
        if isinstance(op, stim.CircuitInstruction):
            flat.append(op)
        elif isinstance(op, stim.CircuitRepeatBlock):
            body = op.body_copy()
            body_flat = _iter_flat_instructions(body)
            for _ in range(op.repeat_count):
                flat.extend(body_flat)
        else:
            raise TypeError(f"Unsupported stim operation type: {type(op)}")
    return flat


class TimelineBuilder:
    """Convert a circuit into serial operation events with explicit times."""

    # Measurement/reset families recognized by name.
    _MEASURE_GATES = {
        "M",
        "MX",
        "MY",
        "MZ",
        "MR",
        "MRX",
        "MRY",
        "MRZ",
        "MPP",
    }
    _RESET_GATES = {"R", "RX", "RY", "RZ"}
    _TWO_QUBIT_GATES = {
        "CNOT",
        "CX",
        "CY",
        "CZ",
        "SWAP",
        "ISWAP",
        "SQRT_XX",
        "SQRT_YY",
        "SQRT_ZZ",
        "XCX",
        "XCY",
        "XCZ",
        "YCX",
        "YCY",
        "YCZ",
        "ZCX",
        "ZCY",
        "ZCZ",
    }

    def __init__(self, durations: GateDurations):
        self.durations = durations

    def duration_for(self, instruction: stim.CircuitInstruction) -> float:
        """Return duration (ns) for one instruction under this schedule."""
        name = instruction.name
        if name in self._MEASURE_GATES:
            return float(self.durations.measure_ns)
        if name in self._RESET_GATES:
            return float(self.durations.reset_ns)
        if name in self._TWO_QUBIT_GATES:
            return float(self.durations.two_qubit_ns)
        # Default: treat as a 1-qubit gate.
        return float(self.durations.one_qubit_ns)

    def build_events(self, circuit: stim.Circuit) -> List[OperationEvent]:
        """Build serial operation events with explicit start/end timestamps."""
        instructions = _iter_flat_instructions(circuit)
        events: List[OperationEvent] = []
        t = 0.0
        for inst in instructions:
            d = self.duration_for(inst)
            start = t
            end = start + d
            events.append(OperationEvent(instruction=inst, start_ns=start, end_ns=end))
            # Serial schedule: next op starts after fixed idle.
            t = end + float(self.durations.idle_ns)
        return events


def integrate_rate(
    fn: RateFn,
    qubit_index: int,
    t_start_ns: float,
    t_end_ns: float,
    steps: int = 8,
) -> float:
    """Numerically integrate an instantaneous rate over [t_start_ns, t_end_ns].

    Uses trapezoidal integration with `steps` samples (>=2 recommended).
    """

    if t_end_ns <= t_start_ns:
        return 0.0
    if steps < 2:
        steps = 2
    ts = np.linspace(t_start_ns, t_end_ns, steps)
    ys = np.array([fn(float(t), int(qubit_index)) for t in ts], dtype=float)
    val = float(np.trapz(ys, ts))
    return val


def validate_pauli_probabilities(
    px: float,
    py: float,
    pz: float,
    tolerance: float = 1e-12,
) -> Tuple[float, float, float]:
    """Validate and sanitize a single-qubit Pauli channel probability triple."""

    vals = np.array([px, py, pz], dtype=float)
    # Clean tiny negative numerical drift.
    vals[np.abs(vals) < tolerance] = 0.0

    if np.any(vals < -tolerance):
        raise ValueError(f"Negative Pauli probability detected: {vals.tolist()}")

    total = float(np.sum(vals))
    if total > 1.0 + tolerance:
        raise ValueError(
            "Invalid Pauli channel: px+py+pz exceeds 1. "
            f"(px,py,pz)=({vals[0]:.6g},{vals[1]:.6g},{vals[2]:.6g}), sum={total:.6g}"
        )
    return float(vals[0]), float(vals[1]), float(vals[2])


class TimeDependentPauliNoiseModel(NoiseModel):
    """Time-dependent, per-qubit Pauli noise injected only during idle windows.

    Parameters are given as instantaneous coefficient/rate functions:
      pX(t, q), pY(t, q), pZ(t, q)
    and are integrated on each idle window to produce a discrete channel.
    """

    def __init__(
        self,
        p_x: RateSpec = 0.0,
        p_y: RateSpec = 0.0,
        p_z: RateSpec = 0.0,
        durations: Optional[GateDurations] = None,
        enabled: bool = True,
        integration_steps: int = 8,
        validate: bool = True,
        tolerance: float = 1e-12,
    ):
        self.enabled = bool(enabled)
        self.p_x_fn = _as_rate_fn(p_x)
        self.p_y_fn = _as_rate_fn(p_y)
        self.p_z_fn = _as_rate_fn(p_z)
        self.durations = durations if durations is not None else GateDurations()
        self.integration_steps = int(integration_steps)
        self.validate = bool(validate)
        self.tolerance = float(tolerance)
        self.timeline_builder = TimelineBuilder(self.durations)

    def _idle_probabilities_for_qubit(
        self,
        qubit_index: int,
        idle_start_ns: float,
        idle_end_ns: float,
    ) -> Tuple[float, float, float]:
        """Compute integrated Pauli probabilities for one qubit in one idle window."""

        px = integrate_rate(
            self.p_x_fn, qubit_index, idle_start_ns, idle_end_ns, self.integration_steps
        )
        py = integrate_rate(
            self.p_y_fn, qubit_index, idle_start_ns, idle_end_ns, self.integration_steps
        )
        pz = integrate_rate(
            self.p_z_fn, qubit_index, idle_start_ns, idle_end_ns, self.integration_steps
        )

        if self.validate:
            px, py, pz = validate_pauli_probabilities(px, py, pz, self.tolerance)

        return px, py, pz

    def _append_idle_noise_window(
        self,
        noisy_circuit: stim.Circuit,
        num_qubits: int,
        idle_start_ns: float,
        idle_end_ns: float,
    ) -> None:
        """Append per-qubit Pauli channels corresponding to one idle interval."""

        for q in range(num_qubits):
            px, py, pz = self._idle_probabilities_for_qubit(
                qubit_index=q,
                idle_start_ns=idle_start_ns,
                idle_end_ns=idle_end_ns,
            )
            if px == 0.0 and py == 0.0 and pz == 0.0:
                continue
            noisy_circuit.append("PAULI_CHANNEL_1", [q], [px, py, pz])

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """Return a new circuit with idle-only time-dependent Pauli noise appended.

        Implementation details:
          - Keep original instruction order.
          - After each instruction except the last, insert one fixed idle window.
          - Inject per-qubit PAULI_CHANNEL_1 for that idle window.
        """

        if not self.enabled:
            return circuit

        events = self.timeline_builder.build_events(circuit)
        if not events:
            return circuit

        noisy = stim.Circuit()
        num_qubits = int(circuit.num_qubits)

        for i, ev in enumerate(events):
            inst = ev.instruction
            noisy.append(inst.name, inst.targets_copy(), inst.gate_args_copy())

            # No idle after final operation by design.
            if i == len(events) - 1:
                continue

            idle_start = ev.end_ns
            idle_end = ev.end_ns + float(self.durations.idle_ns)
            self._append_idle_noise_window(
                noisy_circuit=noisy,
                num_qubits=num_qubits,
                idle_start_ns=idle_start,
                idle_end_ns=idle_end,
            )

        return noisy
