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
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import stim

# Type: f(t_ns, qubit_index) -> instantaneous coefficient/rate.
RateFn = Callable[[float, int], float]
RateSpec = Union[float, RateFn]
GateProbFn = Callable[[int, float, stim.CircuitInstruction], float]
GateProbSpec = Union[float, GateProbFn]


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


def _as_gate_prob_fn(spec: GateProbSpec) -> GateProbFn:
    """Normalize scalar/callable gate-prob spec to a callable."""
    if callable(spec):
        return spec

    constant = float(spec)

    def _const_prob(_op_index: int, _t_ns: float, _inst: stim.CircuitInstruction) -> float:
        return constant

    return _const_prob


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


def validate_probability(p: float, tolerance: float = 1e-12) -> float:
    """Validate and sanitize a scalar probability."""
    val = float(p)
    if abs(val) < tolerance:
        val = 0.0
    if val < -tolerance:
        raise ValueError(f"Negative probability detected: p={val}")
    if val > 1.0 + tolerance:
        raise ValueError(f"Probability exceeds 1: p={val}")
    return float(max(0.0, min(1.0, val)))


def _extract_qubit_targets(instruction: stim.CircuitInstruction) -> List[int]:
    """Best-effort extraction of integer qubit targets from an instruction."""
    qubits: List[int] = []
    for target in instruction.targets_copy():
        # Stim GateTarget usually exposes `.value`.
        if hasattr(target, "value"):
            try:
                qubits.append(int(target.value))
                continue
            except Exception:
                pass
        # Fallback for integer-like targets.
        try:
            qubits.append(int(target))
        except Exception:
            continue
    return qubits


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


class GateDepolarizingNoiseModel(NoiseModel):
    """Inject depolarizing channels directly after 1q/2q gates.

    This model is useful when gate error rates are the control target, as in
    Google-style drift simulations where each gate has an effective
    depolarization probability.
    """

    # Reuse gate families from the timeline builder.
    _TWO_QUBIT_GATES = TimelineBuilder._TWO_QUBIT_GATES
    _MEASURE_GATES = TimelineBuilder._MEASURE_GATES
    _RESET_GATES = TimelineBuilder._RESET_GATES

    def __init__(
        self,
        p_1q: GateProbSpec = 0.0,
        p_2q: GateProbSpec = 0.0,
        enabled: bool = True,
        validate: bool = True,
        tolerance: float = 1e-12,
        apply_on: Literal["unitary_only", "all_non_measure_reset"] = "unitary_only",
    ):
        self.enabled = bool(enabled)
        self.p_1q_fn = _as_gate_prob_fn(p_1q)
        self.p_2q_fn = _as_gate_prob_fn(p_2q)
        self.validate = bool(validate)
        self.tolerance = float(tolerance)
        self.apply_on = apply_on
        self.timeline_builder = TimelineBuilder(GateDurations())

    def _allow_noise_on_instruction(self, name: str) -> bool:
        if self.apply_on == "unitary_only":
            return name not in self._MEASURE_GATES and name not in self._RESET_GATES
        if self.apply_on == "all_non_measure_reset":
            return name not in self._MEASURE_GATES and name not in self._RESET_GATES
        raise ValueError(f"Unknown apply_on mode: {self.apply_on}")

    def _get_p_1q(self, op_index: int, t_ns: float, instruction: stim.CircuitInstruction) -> float:
        p = float(self.p_1q_fn(op_index, t_ns, instruction))
        return validate_probability(p, self.tolerance) if self.validate else p

    def _get_p_2q(self, op_index: int, t_ns: float, instruction: stim.CircuitInstruction) -> float:
        p = float(self.p_2q_fn(op_index, t_ns, instruction))
        return validate_probability(p, self.tolerance) if self.validate else p

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """
        Return circuit with per-gate depolarizing channels appended.
        This is follow the Google's idea: all optimization of gate enters depolarizing noise model, and RL learns to minimize effective depolarization by tuning controls.
        While the Clifford gate is FIXED. 
        """
        if not self.enabled:
            return circuit

        events = self.timeline_builder.build_events(circuit)
        if not events:
            return circuit

        noisy = stim.Circuit()
        for op_index, ev in enumerate(events):
            inst = ev.instruction
            name = inst.name
            noisy.append(name, inst.targets_copy(), inst.gate_args_copy())

            if not self._allow_noise_on_instruction(name):
                continue

            qubits = _extract_qubit_targets(inst)
            if not qubits:
                continue

            if name in self._TWO_QUBIT_GATES:
                p2 = self._get_p_2q(op_index, ev.start_ns, inst)
                if p2 == 0.0:
                    continue
                # DEPOLARIZE2 consumes targets in pairs.
                for k in range(0, len(qubits) - 1, 2):
                    noisy.append("DEPOLARIZE2", [qubits[k], qubits[k + 1]], [p2])
            else:
                p1 = self._get_p_1q(op_index, ev.start_ns, inst)
                if p1 == 0.0:
                    continue
                noisy.append("DEPOLARIZE1", qubits, [p1])

        return noisy


class GoogleLikeDepolarizingNoiseModel(GateDepolarizingNoiseModel):
    """Approximate Google-style drifted gate-error model.

    The model uses:
      p_1q(t) = p_1q_base + sensitivity_1q * agg((u - u_opt(t))^2)
      p_2q(t) = p_2q_base + sensitivity_2q * agg((u - u_opt(t))^2)

    where `u` is the control vector and `u_opt(t)` is a drifting optimum.
    """

    def __init__(
        self,
        control: Sequence[float],
        optimal_control_fn: Callable[[float], Sequence[float]],
        p_1q_base: float,
        p_2q_base: float,
        sensitivity_1q: float,
        sensitivity_2q: float,
        aggregation: Literal["mean", "sum"] = "mean",
        p_clip_max: float = 1.0,
        enabled: bool = True,
        validate: bool = True,
        tolerance: float = 1e-12,
        apply_on: Literal["unitary_only", "all_non_measure_reset"] = "unitary_only",
    ):
        self.control = np.asarray(control, dtype=float)
        self.optimal_control_fn = optimal_control_fn
        self.p_1q_base = float(p_1q_base)
        self.p_2q_base = float(p_2q_base)
        self.sensitivity_1q = float(sensitivity_1q)
        self.sensitivity_2q = float(sensitivity_2q)
        self.aggregation = aggregation
        self.p_clip_max = float(p_clip_max)

        super().__init__(
            p_1q=self._p_1q,
            p_2q=self._p_2q,
            enabled=enabled,
            validate=validate,
            tolerance=tolerance,
            apply_on=apply_on,
        )

    def set_control(self, control: Sequence[float]) -> None:
        """Update control vector, e.g. from RL action."""
        self.control = np.asarray(control, dtype=float)

    def _miscalibration(self, t_ns: float) -> float:
        opt = np.asarray(self.optimal_control_fn(float(t_ns)), dtype=float)
        if opt.shape != self.control.shape:
            raise ValueError(
                "optimal_control_fn output shape mismatch: "
                f"control={self.control.shape}, optimal={opt.shape}"
            )
        delta_sq = np.square(self.control - opt)
        if self.aggregation == "mean":
            return float(np.mean(delta_sq))
        if self.aggregation == "sum":
            return float(np.sum(delta_sq))
        raise ValueError(f"Unknown aggregation mode: {self.aggregation}")

    def _clip_probability(self, p: float) -> float:
        return float(np.clip(p, 0.0, self.p_clip_max))

    def _p_1q(self, _op_index: int, t_ns: float, _instruction: stim.CircuitInstruction) -> float:
        p = self.p_1q_base + self.sensitivity_1q * self._miscalibration(t_ns)
        return self._clip_probability(p)

    def _p_2q(self, _op_index: int, t_ns: float, _instruction: stim.CircuitInstruction) -> float:
        p = self.p_2q_base + self.sensitivity_2q * self._miscalibration(t_ns)
        return self._clip_probability(p)

    def effective_error_rates(self, t_ns: float = 0.0) -> Tuple[float, float]:
        """Return (p_1q, p_2q) at a given time argument."""
        p_1q = self._clip_probability(self.p_1q_base + self.sensitivity_1q * self._miscalibration(t_ns))
        p_2q = self._clip_probability(self.p_2q_base + self.sensitivity_2q * self._miscalibration(t_ns))
        if self.validate:
            p_1q = validate_probability(p_1q, self.tolerance)
            p_2q = validate_probability(p_2q, self.tolerance)
        return p_1q, p_2q


class GoogleLikeGateSpecificNoiseModel(GateDepolarizingNoiseModel):
    """Gate-specific drifted depolarizing model.

    Unlike `GoogleLikeDepolarizingNoiseModel`, which aggregates control mismatch
    into one global scalar, this model maps each gate instruction to a control
    slot and computes gate-local depolarization probabilities.
    """

    def __init__(
        self,
        control: Sequence[float],
        optimal_control_fn: Callable[[float], Sequence[float]],
        p_1q_base: float,
        p_2q_base: float,
        sensitivity_1q: float,
        sensitivity_2q: float,
        n_1q_slots: int,
        n_2q_slots: int,
        p_clip_max: float = 1.0,
        enabled: bool = True,
        validate: bool = True,
        tolerance: float = 1e-12,
        apply_on: Literal["unitary_only", "all_non_measure_reset"] = "unitary_only",
    ):
        self.n_1q_slots = int(n_1q_slots)
        self.n_2q_slots = int(n_2q_slots)
        if self.n_1q_slots <= 0 or self.n_2q_slots <= 0:
            raise ValueError("n_1q_slots and n_2q_slots must be positive.")

        self.control = np.asarray(control, dtype=float).reshape(-1)
        self._control_1q: np.ndarray
        self._control_2q: np.ndarray
        self._slot_cache_1q: Dict[Tuple[str, Tuple[int, ...]], int] = {}
        self._slot_cache_2q: Dict[Tuple[str, Tuple[int, ...]], int] = {}
        self.optimal_control_fn = optimal_control_fn
        self.p_1q_base = float(p_1q_base)
        self.p_2q_base = float(p_2q_base)
        self.sensitivity_1q = float(sensitivity_1q)
        self.sensitivity_2q = float(sensitivity_2q)
        self.p_clip_max = float(p_clip_max)

        expected = self.n_1q_slots + self.n_2q_slots
        if self.control.shape[0] != expected:
            raise ValueError(f"Expected control dim {expected}, got {self.control.shape[0]}")
        self._control_1q, self._control_2q = self._split(self.control)

        super().__init__(
            p_1q=self._p_1q,
            p_2q=self._p_2q,
            enabled=enabled,
            validate=validate,
            tolerance=tolerance,
            apply_on=apply_on,
        )

    def set_control(self, control: Sequence[float]) -> None:
        """Update full control vector."""
        vec = np.asarray(control, dtype=float).reshape(-1)
        expected = self.n_1q_slots + self.n_2q_slots
        if vec.shape[0] != expected:
            raise ValueError(f"Expected control dim {expected}, got {vec.shape[0]}")
        self.control = vec
        self._control_1q, self._control_2q = self._split(self.control)

    def _split(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return vec[: self.n_1q_slots], vec[self.n_1q_slots :]

    def _clip_probability(self, p: float) -> float:
        return float(np.clip(p, 0.0, self.p_clip_max))

    def _stable_slot(self, instruction: stim.CircuitInstruction, n_slots: int) -> int:
        """Deterministic slot mapping from instruction signature."""
        if n_slots == 1:
            return 0
        qubits = _extract_qubit_targets(instruction)
        acc = 2166136261
        for ch in instruction.name.encode("utf-8", errors="ignore"):
            acc = ((acc ^ int(ch)) * 16777619) & 0xFFFFFFFF
        for q in qubits:
            acc = ((acc ^ (int(q) + 31)) * 16777619) & 0xFFFFFFFF
        return int(acc % n_slots)

    def _stable_slot_cached(
        self,
        instruction: stim.CircuitInstruction,
        n_slots: int,
        cache: Dict[Tuple[str, Tuple[int, ...]], int],
    ) -> int:
        if n_slots == 1:
            return 0
        qubits = tuple(_extract_qubit_targets(instruction))
        key = (instruction.name, qubits)
        slot = cache.get(key)
        if slot is None:
            slot = self._stable_slot(instruction, n_slots)
            cache[key] = int(slot)
        return int(slot)

    def _optimal_split(self, t_ns: float) -> Tuple[np.ndarray, np.ndarray]:
        opt = np.asarray(self.optimal_control_fn(float(t_ns)), dtype=float).reshape(-1)
        expected = self.n_1q_slots + self.n_2q_slots
        if opt.shape[0] != expected:
            raise ValueError(
                "optimal_control_fn output shape mismatch: "
                f"control={expected}, optimal={opt.shape[0]}"
            )
        return self._split(opt)

    def _p_1q(self, _op_index: int, t_ns: float, instruction: stim.CircuitInstruction) -> float:
        o1, _ = self._optimal_split(t_ns)
        slot = self._stable_slot_cached(instruction, self.n_1q_slots, self._slot_cache_1q)
        mse = float((self._control_1q[slot] - o1[slot]) ** 2)
        p = self.p_1q_base + self.sensitivity_1q * mse
        return self._clip_probability(p)

    def _p_2q(self, _op_index: int, t_ns: float, instruction: stim.CircuitInstruction) -> float:
        _, o2 = self._optimal_split(t_ns)
        slot = self._stable_slot_cached(instruction, self.n_2q_slots, self._slot_cache_2q)
        mse = float((self._control_2q[slot] - o2[slot]) ** 2)
        p = self.p_2q_base + self.sensitivity_2q * mse
        return self._clip_probability(p)

    def effective_error_rates(self, t_ns: float = 0.0) -> Tuple[float, float]:
        """Return mean (p_1q, p_2q) implied by slot-wise mismatch."""
        o1, o2 = self._optimal_split(t_ns)
        mse_1q = float(np.mean(np.square(self._control_1q - o1)))
        mse_2q = float(np.mean(np.square(self._control_2q - o2)))
        p_1q = self._clip_probability(self.p_1q_base + self.sensitivity_1q * mse_1q)
        p_2q = self._clip_probability(self.p_2q_base + self.sensitivity_2q * mse_2q)
        if self.validate:
            p_1q = validate_probability(p_1q, self.tolerance)
            p_2q = validate_probability(p_2q, self.tolerance)
        return p_1q, p_2q
