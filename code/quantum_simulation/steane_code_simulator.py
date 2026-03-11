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


////////////////////////////////////////////////
结论分两层：
  1. 单次 step 的测量结果确实是一次随机样本
  - 不是“真值”。
  - 只看一个 step 的一个 bit，证据很弱。
  2. QEC 不靠单个 bit 下结论，而是靠时空冗余
  - 空间上：同一轮会看 6 个 stabilizer。
  - 时间上：会跨多轮比较 syndrome（你这里 60 steps = 10 rounds）。
  - 你脚本里 MV/DE 就是在做这种时间聚合（多数票或相邻差分）。
  所以你说“不能仅凭一次测量”完全正确。
  当前代码也是这个思路，只是：
  - noise 现在多半是无噪声/很弱时，很多结果看起来“太干净”；
  - 一旦加入 realistic 噪声，单 step 波动会明显，必须依赖多轮聚合和 decoder。
////////////////////////////////////////////////////////////////////////

"""

from __future__ import annotations

import pickle
import argparse
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
import os
from pathlib import Path
import sys
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np
import stim
try:
    from noise_engine import (
        ComposedGateAndCorrelatedIdleNoiseModel,
        HiddenMarkovCorrelatedPauliNoiseModel,
        NoiseModel,
        PreMeasurementBitFlipNoiseModel,
    )
except ModuleNotFoundError:
    # Package import path, e.g. `from quantum_simulation.steane_code_simulator import ...`
    from quantum_simulation.noise_engine import (
        ComposedGateAndCorrelatedIdleNoiseModel,
        HiddenMarkovCorrelatedPauliNoiseModel,
        NoiseModel,
        PreMeasurementBitFlipNoiseModel,
    )


LOGGER = logging.getLogger(__name__)
_ENABLE_STREAMING_NOISE = os.environ.get("STEANE_DISABLE_STREAMING_NOISE", "0") != "1"

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

# Note for collaborators:
# - Steane workflow code stays in this file.
# - Noise/timeline logic is intentionally decoupled in `noise_engine.py`.
# - You can replace the default `NoiseModel` with
#   `TimeDependentPauliNoiseModel` from that module without changing the
#   experiment flow code below.


class _ApplyCacheNoiseModel(NoiseModel):
    """Thin wrapper that memoizes `noise.apply(circuit)` by circuit content.

    In RL/Monte-Carlo loops, the same few sub-circuits are repeatedly passed to
    the same noise model instance across many shots. Caching the transformed
    noisy circuit avoids re-running expensive timeline/noise insertion logic.
    """

    def __init__(self, base: NoiseModel):
        self.base = base
        self.enabled = bool(getattr(base, "enabled", True))
        self.disable_apply_cache = bool(getattr(base, "disable_apply_cache", False))
        self.stateful = bool(getattr(base, "stateful", False))
        self.supports_parallel_shots = bool(getattr(base, "supports_parallel_shots", False))
        self._cache: dict[str, stim.Circuit] = {}

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        if not self.enabled:
            return circuit
        key = str(circuit)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        out = self.base.apply(circuit)
        self._cache[key] = out
        return out

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def fork_for_shot(self, shot_index: int) -> NoiseModel:
        """Fork base model first, then keep apply-cache behavior per shot."""
        fork_fn = getattr(self.base, "fork_for_shot", None)
        if callable(fork_fn):
            return _ApplyCacheNoiseModel(fork_fn(int(shot_index)))
        return self


def _parallel_noise_for_shot(noise: NoiseModel, shot_index: int) -> NoiseModel:
    """Return shot-local noise instance when model supports safe parallel forking."""
    if not bool(getattr(noise, "stateful", False)):
        return noise
    if not bool(getattr(noise, "supports_parallel_shots", False)):
        return noise
    fork_fn = getattr(noise, "fork_for_shot", None)
    if not callable(fork_fn):
        return noise
    return fork_fn(int(shot_index))


@dataclass(frozen=True)
class _StreamEventPlan:
    pre_instruction_noise: tuple[stim.CircuitInstruction, ...]
    ideal_instruction: stim.CircuitInstruction
    gate_noise_instructions: tuple[stim.CircuitInstruction, ...]
    add_idle_window: bool


@dataclass(frozen=True)
class _StreamCircuitPlan:
    events: tuple[_StreamEventPlan, ...]
    num_qubits: int


def _extract_qubit_targets_local(instruction: stim.CircuitInstruction) -> list[int]:
    qubits: list[int] = []
    for target in instruction.targets_copy():
        if hasattr(target, "value"):
            try:
                qubits.append(int(target.value))
                continue
            except Exception:
                pass
        try:
            qubits.append(int(target))
        except Exception:
            continue
    return qubits


def _make_single_instruction(
    name: str,
    targets: Sequence[int],
    gate_args: Sequence[float] = (),
) -> stim.CircuitInstruction:
    # Construct instruction directly to avoid per-call temporary Circuit+append overhead.
    return stim.CircuitInstruction(
        str(name),
        [int(t) for t in targets],
        [float(x) for x in gate_args],
    )


def _compile_stream_circuit_plan(
    noise: NoiseModel,
    circuit: stim.Circuit,
) -> Optional[_StreamCircuitPlan]:
    measurement_model: Any = None
    base_noise = noise
    if isinstance(noise, PreMeasurementBitFlipNoiseModel):
        measurement_model = noise
        base_noise = noise.base_model

    gate_model: Any = None
    idle_model: Any = None
    if isinstance(base_noise, HiddenMarkovCorrelatedPauliNoiseModel):
        idle_model = base_noise
        timeline_builder = base_noise.timeline_builder
    elif isinstance(base_noise, ComposedGateAndCorrelatedIdleNoiseModel):
        gate_model = base_noise.gate_model
        idle_model = base_noise.idle_model
        timeline_builder = base_noise.timeline_builder
    else:
        return None

    if not hasattr(idle_model, "apply_idle_window_to_simulator"):
        return None
    events = timeline_builder.build_events(circuit)
    if not events:
        return _StreamCircuitPlan(events=tuple(), num_qubits=int(circuit.num_qubits))

    out_events: list[_StreamEventPlan] = []
    idle_enabled = bool(getattr(idle_model, "enabled", True))
    for op_index, ev in enumerate(events):
        inst = ev.instruction
        pre_instruction_noise: list[stim.CircuitInstruction] = []
        gate_noise_instructions: list[stim.CircuitInstruction] = []

        if measurement_model is not None and bool(getattr(measurement_model, "enabled", True)):
            if measurement_model._should_inject_before(inst):
                qubits = _extract_qubit_targets_local(inst)
                if qubits and float(measurement_model.p_flip) > 0.0:
                    pre_instruction_noise.append(
                        _make_single_instruction("X_ERROR", qubits, [float(measurement_model.p_flip)])
                    )

        if gate_model is not None and bool(getattr(gate_model, "enabled", True)):
            name = inst.name
            if gate_model._allow_noise_on_instruction(name):
                qubits = _extract_qubit_targets_local(inst)
                if qubits:
                    if name in gate_model._TWO_QUBIT_GATES:
                        p2 = float(gate_model._get_p_2q(op_index, ev.start_ns, inst))
                        if p2 > 0.0:
                            pair_targets = qubits[: (len(qubits) // 2) * 2]
                            if pair_targets:
                                gate_noise_instructions.append(
                                    _make_single_instruction("DEPOLARIZE2", pair_targets, [p2])
                                )
                    else:
                        p1 = float(gate_model._get_p_1q(op_index, ev.start_ns, inst))
                        if p1 > 0.0:
                            gate_noise_instructions.append(
                                _make_single_instruction("DEPOLARIZE1", qubits, [p1])
                            )

        out_events.append(
            _StreamEventPlan(
                pre_instruction_noise=tuple(pre_instruction_noise),
                ideal_instruction=inst,
                gate_noise_instructions=tuple(gate_noise_instructions),
                add_idle_window=(op_index < len(events) - 1 and idle_enabled),
            )
        )

    return _StreamCircuitPlan(events=tuple(out_events), num_qubits=int(circuit.num_qubits))


def _sim_do_with_noise(
    simulator: stim.TableauSimulator,
    noise: NoiseModel,
    circuit: stim.Circuit,
    stream_plan_cache: Optional[dict[tuple[int, int], tuple[stim.Circuit, _StreamCircuitPlan]]] = None,
) -> None:
    if _ENABLE_STREAMING_NOISE and stream_plan_cache is not None:
        stream_noise = noise.base_model if isinstance(noise, PreMeasurementBitFlipNoiseModel) else noise
        measure_key = 0
        if isinstance(noise, PreMeasurementBitFlipNoiseModel):
            measure_key = int(round(float(noise.p_flip) * 1e12))
        if isinstance(stream_noise, HiddenMarkovCorrelatedPauliNoiseModel):
            noise_key = hash(("corr_idle", measure_key))
        elif isinstance(stream_noise, ComposedGateAndCorrelatedIdleNoiseModel):
            noise_key = hash(("composed_corr_idle", int(id(stream_noise.gate_model)), measure_key))
        else:
            noise_key = hash(("generic_noise", int(id(noise)), measure_key))
        cache_key = (int(id(circuit)), int(noise_key))
        cached = stream_plan_cache.get(cache_key)
        plan = None
        if cached is not None and cached[0] is circuit:
            plan = cached[1]
        else:
            compiled = _compile_stream_circuit_plan(noise=noise, circuit=circuit)
            if compiled is not None:
                stream_plan_cache[cache_key] = (circuit, compiled)
                plan = compiled

        if plan is not None:
            idle_model = stream_noise if isinstance(stream_noise, HiddenMarkovCorrelatedPauliNoiseModel) else getattr(
                stream_noise,
                "idle_model",
                None,
            )
            if idle_model is None:
                simulator.do(noise.apply(circuit))
                return
            apply_idle_window = getattr(idle_model, "apply_idle_window_to_simulator", None)
            if not callable(apply_idle_window):
                simulator.do(noise.apply(circuit))
                return

            for event in plan.events:
                for noise_inst in event.pre_instruction_noise:
                    simulator.do(noise_inst)
                simulator.do(event.ideal_instruction)
                for noise_inst in event.gate_noise_instructions:
                    simulator.do(noise_inst)
                if event.add_idle_window:
                    apply_idle_window(simulator, int(plan.num_qubits))
            return

    simulator.do(noise.apply(circuit))


def _run_shot_batch(
    shot_indices: Sequence[int],
    initial_state: str,
    meas_basis: str,
    n_steps: int,
    syndrome_mode: Literal["MV", "DE"],
    noise: NoiseModel,
    encoding_circuit_tpl: stim.Circuit,
    prep_circuit_tpl: stim.Circuit,
    stabilizer_circuit_tpls: Sequence[stim.Circuit],
    rotate_circuit_tpl: stim.Circuit,
    logical_measurement_circuit_tpl: stim.Circuit,
) -> list[tuple[int, int]]:
    """Run one batch of shots, returning `(shot_idx, success)` pairs."""
    out: list[tuple[int, int]] = []
    stream_plan_cache: dict[tuple[int, int], tuple[stim.Circuit, _StreamCircuitPlan]] = {}
    for shot_idx in shot_indices:
        shot_noise = _parallel_noise_for_shot(noise, int(shot_idx))
        success, _ = _run_single_shot_sequential(
            initial_state=initial_state,
            meas_basis=meas_basis,
            n_steps=int(n_steps),
            syndrome_mode=syndrome_mode,
            noise=shot_noise,
            save_trace=False,
            encoding_circuit_tpl=encoding_circuit_tpl,
            prep_circuit_tpl=prep_circuit_tpl,
            stabilizer_circuit_tpls=stabilizer_circuit_tpls,
            rotate_circuit_tpl=rotate_circuit_tpl,
            logical_measurement_circuit_tpl=logical_measurement_circuit_tpl,
            stream_plan_cache=stream_plan_cache,
        )
        out.append((int(shot_idx), int(success)))
    return out


def encoding_circuit(log_qb_idx: int = 0) -> stim.Circuit:
    """
    构建 Steane 逻辑 |0>_L 编码电路（含 ancilla 验证）。 

    别的逻辑态，是在它后面再接逻辑门（或 prepare_stab_eigenstate(...)）

    电路结构沿用 tutorial 的 Goto-style 方案：先制备 7 个数据比特的编码态，
    再用该逻辑块的第 8 个物理位（索引 7+s）测一次逻辑 Z_L 做验收。

    Args:
        log_qb_idx: 逻辑比特编号。每个逻辑块占 8 个物理位，因此会用 `s=8*idx`
            做索引平移。

    Returns:
        stim.Circuit: 编码+验证电路。
    """
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
    """
    对指定逻辑块施加横向（transversal）逻辑单比特门。

    支持门：
        - "X", "Z": 按 tutorial 中选定的逻辑算符支撑位作用在 4/5/6 三个数据位
        - "H", "S", "S_DAG": 作用在该逻辑块全部 7 个数据位

    Args:
        gate: 逻辑门名称。
        log_qb_idx: 目标逻辑比特编号（用于 8*idx 偏移）。

    Returns:
        stim.Circuit: 对应的逻辑门电路。
    """
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
    """
    从已编码 |0>_L 出发，制备目标逻辑 Pauli 本征态。

    约定：
        +Z/-Z/+X/-X/+Y/-Y 分别对应 |0>_L, |1>_L, |+>_L, |->_L, |+i>_L, |-i>_L。

    Args:
        stabilizer: 目标本征态标签（例如 "+Z"、"-X"）。

    Returns:
        stim.Circuit: 从 |0>_L 到目标态的逻辑门序列。
    """
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
    """
    构建测量前旋转电路，把目标基测量化为最终物理 Z 测量。

    思路与 tutorial 相同：最终总是测物理 Z，通过逻辑旋转实现 X/Y/Z 三基测量。

    Args:
        meas_basis: 目标测量基，取值 "X"/"Y"/"Z"。

    Returns:
        stim.Circuit: 测量前旋转电路。
    """
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
    """
    破坏性测量一个或多个逻辑块的 7 个数据位（物理 Z 基）。

    注意：该函数不是“直接测逻辑算符门”，而是先测物理位，再在经典后处理中
    通过异或重构逻辑可观测量（例如 Z_L）。

    Args:
        log_qubit_indices: 要测量的逻辑块编号列表；默认只测第 0 个逻辑块。

    Returns:
        stim.Circuit: 对应的物理测量电路。
    """
    if log_qubit_indices is None:
        log_qubit_indices = [0]

    c = stim.Circuit()
    for log_qubit_index in log_qubit_indices:
        s = log_qubit_index * 8
        c.append("M", np.array([0, 1, 2, 3, 4, 5, 6]) + s)
    return c


def measure_single_stabilizer(stab: dict, ancilla: int = 8) -> stim.Circuit:
    """
    使用单个 ancilla 测量一个稳定子，并在末尾重置 ancilla。

    Args:
        stab: 稳定子描述字典，至少包含：
            - `type`: "X" 或 "Z"
            - `data`: 参与该稳定子的物理数据位索引列表
        ancilla: 用作 syndrome 提取的 ancilla 物理索引。

    Returns:
        stim.Circuit: 单稳定子测量子电路（含 M 和 R）。
    """
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
    """
    多数投票聚合。

    Args:
        bits: 同一稳定子的重复测量历史（0/1）。

    Returns:
        票数更多的结果；空列表返回 0。平票时返回 0（偏向无事件）。
    """
    if not bits:
        return 0
    ones = int(np.sum(bits))
    zeros = len(bits) - ones
    return int(ones > zeros)


def _detect_event(bits: list[int]) -> int:
    """
    检测事件（Detection Event）聚合。

    使用该稳定子最近两次测量结果的异或作为事件指示。

    Args:
        bits: 同一稳定子的重复测量历史（0/1）。

    Returns:
        最后两次结果是否变化（0/1）；不足两次时返回 0。
    """
    if len(bits) < 2:
        return 0
    return int(bits[-1] ^ bits[-2])


def aggregate_syndromes(histories: dict[str, list[int]], mode: Literal["MV", "DE"]) -> tuple[np.ndarray, np.ndarray]:
    """
    把重复测量历史聚合为当前有效 X/Z syndrome 向量。

    Args:
        histories: 每个稳定子的历史读数字典，例如
            `{ "S1":[...], "S2":[...], ..., "S6":[...] }`。
        mode:
            - "MV": 多数投票
            - "DE": 最近两次差分（detection event）

    Returns:
        `(x, z)` 两个长度为 3 的 `np.ndarray`，分别对应 X/Z 三个稳定子。
    """
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
    """
    tutorial 同款 LUT 解码器（单通道：X 或 Z）。

    Args:
        syndromes: 长度 3 的 syndrome 差分向量。

    Returns:
        1 表示检测到需要在 Pauli frame 中记录一次逻辑翻转；否则 0。
    """
    bad_syndrome_patterns = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1]])
    if np.any(np.all(bad_syndrome_patterns == syndromes, axis=1)):
        LOGGER.debug("Decoder: logical error detected from syndrome=%s", syndromes)
        return 1
    return 0


def expected_result(measure_output: int, initial_state: str, meas_basis: str) -> int:
    """
    判断一次逻辑测量结果是否符合理论期望。

    仅在 `initial_state` 与 `meas_basis` 对易时有效；反对易情形测量本应随机，
    会抛出异常提醒。

    Args:
        measure_output: 单次逻辑测量 bit（0/1）。
        initial_state: 初态标签（如 +Z/-X/+Y）。
        meas_basis: 逻辑测量基（X/Y/Z）。

    Returns:
        1 表示该 shot 命中期望结果，0 表示未命中。
    """
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
    logical_measurement_circuit: Optional[stim.Circuit] = None,
    stream_plan_cache: Optional[dict[tuple[int, int], tuple[stim.Circuit, _StreamCircuitPlan]]] = None,
) -> tuple[int, int]:
    """
    执行最终破坏性测量并输出逻辑测量 bit。

    过程：
        1) 测量 7 个物理数据位
        2) 重构逻辑可观测量与末轮 syndrome
        3) 做一次末轮 unflagged 解码修正
        4) 叠加当前 Pauli frame 得到最终逻辑输出

    Args:
        simulator: 当前 TableauSimulator。
        meas_basis: 目标逻辑测量基（X/Y/Z）。
        tracked_x_syndromes: 过程跟踪的 X syndrome（长度 3）。
        tracked_z_syndromes: 过程跟踪的 Z syndrome（长度 3）。
        pauli_frame: 当前 Pauli frame（[X_L, Z_L] 两位）。
        m_idx: measurement record 读取偏移。
        noise: 噪声模型对象。

    Returns:
        `(logical_measurement_bit, updated_m_idx)`。
    """
    if logical_measurement_circuit is None:
        logical_measurement_circuit = measure_logical_qubits()
    _sim_do_with_noise(
        simulator=simulator,
        noise=noise,
        circuit=logical_measurement_circuit,
        stream_plan_cache=stream_plan_cache,
    )
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


def _run_single_shot_sequential(
    initial_state: str,
    meas_basis: str,
    n_steps: int,
    syndrome_mode: Literal["MV", "DE"],
    noise: NoiseModel,
    save_trace: bool = False,
    encoding_circuit_tpl: Optional[stim.Circuit] = None,
    prep_circuit_tpl: Optional[stim.Circuit] = None,
    stabilizer_circuit_tpls: Optional[Sequence[stim.Circuit]] = None,
    rotate_circuit_tpl: Optional[stim.Circuit] = None,
    logical_measurement_circuit_tpl: Optional[stim.Circuit] = None,
    stream_plan_cache: Optional[dict[tuple[int, int], tuple[stim.Circuit, _StreamCircuitPlan]]] = None,
) -> tuple[int, Optional[dict[str, Any]]]:
    """运行一个 shot，并可选保存中间轨迹。"""
    if hasattr(noise, "start_shot") and callable(getattr(noise, "start_shot")):
        noise.start_shot()

    if encoding_circuit_tpl is None:
        encoding_circuit_tpl = encoding_circuit()
    if prep_circuit_tpl is None:
        prep_circuit_tpl = prepare_stab_eigenstate(initial_state)
    if stabilizer_circuit_tpls is None:
        stabilizer_circuit_tpls = tuple(
            measure_single_stabilizer(stab, ancilla=8) for stab in STABILIZER_SEQUENCE
        )
    if rotate_circuit_tpl is None:
        rotate_circuit_tpl = rotate_to_measurement_basis(meas_basis)
    if logical_measurement_circuit_tpl is None:
        logical_measurement_circuit_tpl = measure_logical_qubits()

    simulator = stim.TableauSimulator()
    m_idx = 0

    prep_ancilla_measurements: list[int] = []
    step_measurements: list[dict[str, int]] = []
    decoder_updates: list[dict[str, Any]] = []

    prep_ok = False
    for _attempt in range(3):
        _sim_do_with_noise(
            simulator=simulator,
            noise=noise,
            circuit=encoding_circuit_tpl,
            stream_plan_cache=stream_plan_cache,
        )
        state_prep_ancilla = int(simulator.current_measurement_record()[m_idx])
        m_idx += 1
        prep_ancilla_measurements.append(state_prep_ancilla)
        if state_prep_ancilla == 0:
            prep_ok = True
            break

    if not prep_ok:
        if save_trace:
            trace = {
                "prep_ok": False,
                "prep_ancilla_measurements": prep_ancilla_measurements,
                "step_measurements": [],
                "histories": {s["name"]: [] for s in STABILIZER_SEQUENCE},
                "decoder_updates": [],
                "final_measurement": None,
                "final_tracked_x_syndromes": [0, 0, 0],
                "final_tracked_z_syndromes": [0, 0, 0],
                "final_pauli_frame": [0, 0],
                "success": 0,
            }
            return 0, trace
        return 0, None

    _sim_do_with_noise(
        simulator=simulator,
        noise=noise,
        circuit=prep_circuit_tpl,
        stream_plan_cache=stream_plan_cache,
    )

    histories: dict[str, list[int]] = {s["name"]: [] for s in STABILIZER_SEQUENCE}
    tracked_x_syndromes = np.zeros(3, dtype=int)
    tracked_z_syndromes = np.zeros(3, dtype=int)
    pauli_frame = np.array([0, 0], dtype=int)

    for step in range(n_steps):
        stab_idx = step % 6
        stab = STABILIZER_SEQUENCE[stab_idx]
        _sim_do_with_noise(
            simulator=simulator,
            noise=noise,
            circuit=stabilizer_circuit_tpls[stab_idx],
            stream_plan_cache=stream_plan_cache,
        )
        meas = int(simulator.current_measurement_record()[m_idx])
        m_idx += 1
        histories[stab["name"]].append(meas)

        if save_trace:
            step_measurements.append(
                {
                    "step": int(step),
                    "stabilizer_index": int(step % 6),
                    "stabilizer_name": stab["name"],
                    "measurement": int(meas),
                }
            )

        if step % 6 == 5:
            current_x, current_z = aggregate_syndromes(histories, syndrome_mode)
            diff_x = tracked_x_syndromes ^ current_x
            diff_z = tracked_z_syndromes ^ current_z
            pf_before = pauli_frame.copy()
            pf_x = unflagged_decoder(diff_x)
            pf_z = unflagged_decoder(diff_z)
            pauli_frame[0] ^= pf_x
            pauli_frame[1] ^= pf_z
            tracked_x_syndromes = current_x
            tracked_z_syndromes = current_z

            if save_trace:
                decoder_updates.append(
                    {
                        "kind": "full_round",
                        "round_index": int(step // 6),
                        "current_x_syndromes": current_x.astype(int).tolist(),
                        "current_z_syndromes": current_z.astype(int).tolist(),
                        "diff_x": diff_x.astype(int).tolist(),
                        "diff_z": diff_z.astype(int).tolist(),
                        "pf_before": pf_before.astype(int).tolist(),
                        "decoder_flip_x": int(pf_x),
                        "decoder_flip_z": int(pf_z),
                        "pf_after": pauli_frame.astype(int).tolist(),
                    }
                )

    if n_steps % 6 != 0:
        current_x, current_z = aggregate_syndromes(histories, syndrome_mode)
        diff_x = tracked_x_syndromes ^ current_x
        diff_z = tracked_z_syndromes ^ current_z
        pf_before = pauli_frame.copy()
        pf_x = unflagged_decoder(diff_x)
        pf_z = unflagged_decoder(diff_z)
        pauli_frame[0] ^= pf_x
        pauli_frame[1] ^= pf_z
        tracked_x_syndromes = current_x
        tracked_z_syndromes = current_z

        if save_trace:
            decoder_updates.append(
                {
                    "kind": "partial_round",
                    "round_index": int(n_steps // 6),
                    "current_x_syndromes": current_x.astype(int).tolist(),
                    "current_z_syndromes": current_z.astype(int).tolist(),
                    "diff_x": diff_x.astype(int).tolist(),
                    "diff_z": diff_z.astype(int).tolist(),
                    "pf_before": pf_before.astype(int).tolist(),
                    "decoder_flip_x": int(pf_x),
                    "decoder_flip_z": int(pf_z),
                    "pf_after": pauli_frame.astype(int).tolist(),
                }
            )

    _sim_do_with_noise(
        simulator=simulator,
        noise=noise,
        circuit=rotate_circuit_tpl,
        stream_plan_cache=stream_plan_cache,
    )
    final_measurement, m_idx = destructive_logical_measurement(
        simulator=simulator,
        meas_basis=meas_basis,
        tracked_x_syndromes=tracked_x_syndromes,
        tracked_z_syndromes=tracked_z_syndromes,
        pauli_frame=pauli_frame,
        m_idx=m_idx,
        noise=noise,
        logical_measurement_circuit=logical_measurement_circuit_tpl,
        stream_plan_cache=stream_plan_cache,
    )

    success = int(expected_result(final_measurement, initial_state, meas_basis))

    if not save_trace:
        return success, None

    trace = {
        "prep_ok": True,
        "prep_ancilla_measurements": prep_ancilla_measurements,
        "step_measurements": step_measurements,
        "histories": {k: [int(x) for x in v] for k, v in histories.items()},
        "decoder_updates": decoder_updates,
        "final_measurement": int(final_measurement),
        "final_tracked_x_syndromes": tracked_x_syndromes.astype(int).tolist(),
        "final_tracked_z_syndromes": tracked_z_syndromes.astype(int).tolist(),
        "final_pauli_frame": pauli_frame.astype(int).tolist(),
        "success": success,
    }
    return success, trace


@lru_cache(maxsize=32)
def _cached_shot_templates(
    initial_state: str,
    meas_basis: str,
) -> tuple[stim.Circuit, stim.Circuit, tuple[stim.Circuit, ...], stim.Circuit, stim.Circuit]:
    """Build immutable circuit templates once per (initial_state, meas_basis)."""
    return (
        encoding_circuit(),
        prepare_stab_eigenstate(initial_state),
        tuple(measure_single_stabilizer(stab, ancilla=8) for stab in STABILIZER_SEQUENCE),
        rotate_to_measurement_basis(meas_basis),
        measure_logical_qubits(),
    )


def steane_code_exp_sequential(
    initial_state: str = "+Z",
    meas_basis: str = "Z",
    n_steps: int = 60,
    shots: int = 100,
    syndrome_mode: Literal["MV", "DE"] = "MV",
    noise: Optional[NoiseModel] = None,
    shot_workers: int = 1,
) -> list[int]:
    """
    运行顺序稳定子调度的端到端 Steane QEC 实验。

    调度固定为 `S1 -> ... -> S6 -> ...`，总共执行 `n_steps` 步。
    每走完一个完整 6 步块，就把历史 syndrome 依据 `mode` 聚合并更新 Pauli frame。

    Args:
        initial_state: 逻辑初态（+Z/-Z/+X/-X/+Y/-Y）。
        meas_basis: 最终测量基（X/Y/Z）。
        n_steps: 稳定子测量总步数。
        shots: 统计次数。
        syndrome_mode: 历史聚合方式（MV 或 DE）。
        noise: 噪声模型；默认无噪声。

    Returns:
        `list[int]`：每个 shot 的成功标志（1 成功，0 失败）。
    """
    if noise is None:
        noise = NoiseModel(enabled=False)
    if getattr(noise, "enabled", False) and not bool(getattr(noise, "disable_apply_cache", False)):
        noise = _ApplyCacheNoiseModel(noise)

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    # Build immutable templates once per basis/state pair and reuse across runs.
    (
        encoding_circuit_tpl,
        prep_circuit_tpl,
        stabilizer_circuit_tpls,
        rotate_circuit_tpl,
        logical_measurement_circuit_tpl,
    ) = _cached_shot_templates(str(initial_state), str(meas_basis))

    requested_workers = int(shot_workers)
    if requested_workers > 1 and bool(getattr(noise, "stateful", False)):
        # Stateful models need per-shot independent instances in parallel mode.
        can_parallelize_stateful = bool(getattr(noise, "supports_parallel_shots", False)) and callable(
            getattr(noise, "fork_for_shot", None)
        )
        if not can_parallelize_stateful:
            LOGGER.warning(
                "Stateful noise model does not support safe shot forking; "
                "falling back to shot_workers=1."
            )
            requested_workers = 1

    # Fast single-thread path keeps overhead minimal for tiny shot counts.
    if int(requested_workers) <= 1 or int(shots) <= 1:
        results: list[int] = []
        stream_plan_cache: dict[tuple[int, int], tuple[stim.Circuit, _StreamCircuitPlan]] = {}
        for _ in range(shots):
            success, _ = _run_single_shot_sequential(
                initial_state=initial_state,
                meas_basis=meas_basis,
                n_steps=n_steps,
                syndrome_mode=syndrome_mode,
                noise=noise,
                save_trace=False,
                encoding_circuit_tpl=encoding_circuit_tpl,
                prep_circuit_tpl=prep_circuit_tpl,
                stabilizer_circuit_tpls=stabilizer_circuit_tpls,
                rotate_circuit_tpl=rotate_circuit_tpl,
                logical_measurement_circuit_tpl=logical_measurement_circuit_tpl,
                stream_plan_cache=stream_plan_cache,
            )
            results.append(int(success))
        return results

    # Stateful channels with safe forking are process-parallelized to avoid GIL
    # bottlenecks in Python-heavy noise insertion loops.
    use_process_parallel = bool(getattr(noise, "stateful", False)) and bool(
        getattr(noise, "supports_parallel_shots", False)
    )
    if use_process_parallel:
        # Spawn-based process pools fail in interactive `<stdin>` contexts.
        main_mod = sys.modules.get("__main__")
        main_file = getattr(main_mod, "__file__", "")
        if not main_file or str(main_file) == "<stdin>":
            LOGGER.warning(
                "Process-based shot parallelism unavailable in interactive "
                "session; falling back to thread workers."
            )
            use_process_parallel = False
        else:
            # If model is not pickleable, process workers cannot receive it.
            try:
                pickle.dumps(noise)
            except Exception as exc:
                LOGGER.warning(
                    "Stateful noise model not pickleable (%s); falling back "
                    "to thread workers.",
                    type(exc).__name__,
                )
                use_process_parallel = False
    n_workers = max(1, min(int(requested_workers), int(shots)))
    shot_ids = list(range(int(shots)))
    if use_process_parallel:
        batch_size = max(1, (len(shot_ids) + n_workers - 1) // n_workers)
        batches = [shot_ids[i : i + batch_size] for i in range(0, len(shot_ids), batch_size)]
        results_by_shot: dict[int, int] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(
                    _run_shot_batch,
                    batch,
                    initial_state,
                    meas_basis,
                    n_steps,
                    syndrome_mode,
                    noise,  # type: ignore[arg-type]
                    encoding_circuit_tpl,
                    prep_circuit_tpl,
                    stabilizer_circuit_tpls,
                    rotate_circuit_tpl,
                    logical_measurement_circuit_tpl,
                )
                for batch in batches
            ]
            for fut in as_completed(futures):
                for shot_idx, success in fut.result():
                    results_by_shot[int(shot_idx)] = int(success)
        return [int(results_by_shot[i]) for i in shot_ids]

    # Stateless channels keep thread-level shot parallelism to avoid process
    # spawn/pickle overhead.
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _run_single_shot_sequential,
                initial_state,
                meas_basis,
                n_steps,
                syndrome_mode,
                noise,  # type: ignore[arg-type]
                False,
                encoding_circuit_tpl,
                prep_circuit_tpl,
                stabilizer_circuit_tpls,
                rotate_circuit_tpl,
                logical_measurement_circuit_tpl,
                None,
            )
            for _ in shot_ids
        ]
        return [int(f.result()[0]) for f in futures]


def steane_code_exp_sequential_with_trace(
    initial_state: str = "+Z",
    meas_basis: str = "Z",
    n_steps: int = 60,
    shots: int = 100,
    syndrome_mode: Literal["MV", "DE"] = "MV",
    noise: Optional[NoiseModel] = None,
    shot_workers: int = 1,
) -> dict[str, Any]:
    """运行实验并返回每个 shot 的中间探测轨迹。"""
    if noise is None:
        noise = NoiseModel(enabled=False)
    if getattr(noise, "enabled", False) and not bool(getattr(noise, "disable_apply_cache", False)):
        noise = _ApplyCacheNoiseModel(noise)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    # Build immutable circuit templates once and reuse across all shots.
    (
        encoding_circuit_tpl,
        prep_circuit_tpl,
        stabilizer_circuit_tpls,
        rotate_circuit_tpl,
        logical_measurement_circuit_tpl,
    ) = _cached_shot_templates(str(initial_state), str(meas_basis))

    # Keep trace mode single-threaded for determinism and easier debugging.
    # Parallel trace collection is possible but tends to complicate ordering and
    # reproducibility; RL training should use the non-trace fast path anyway.
    _ = shot_workers
    results: list[int] = []
    traces: list[dict[str, Any]] = []
    stream_plan_cache: dict[tuple[int, int], tuple[stim.Circuit, _StreamCircuitPlan]] = {}
    for shot_index in range(shots):
        success, trace = _run_single_shot_sequential(
            initial_state=initial_state,
            meas_basis=meas_basis,
            n_steps=n_steps,
            syndrome_mode=syndrome_mode,
            noise=noise,
            save_trace=True,
            encoding_circuit_tpl=encoding_circuit_tpl,
            prep_circuit_tpl=prep_circuit_tpl,
            stabilizer_circuit_tpls=stabilizer_circuit_tpls,
            rotate_circuit_tpl=rotate_circuit_tpl,
            logical_measurement_circuit_tpl=logical_measurement_circuit_tpl,
            stream_plan_cache=stream_plan_cache,
        )
        results.append(int(success))
        trace = trace if trace is not None else {}
        trace["shot_index"] = int(shot_index)
        traces.append(trace)

    return {
        "initial_state": initial_state,
        "meas_basis": meas_basis,
        "n_steps": int(n_steps),
        "shots": int(shots),
        "syndrome_mode": syndrome_mode,
        "results": results,
        "traces": traces,
        "success_rate": float(np.mean(results)) if results else 0.0,
    }


class SteaneQECSimulator:
    """Steane QEC 高层封装类。

    目标是让你像 tutorial 一样分块调试：
        - 单独构建某段电路并画图
        - 单独做 sanity-check 验证
        - 一键运行完整实验并拿到统计结果
    """

    def __init__(self, noise: Optional[NoiseModel] = None):
        """初始化模拟器对象。

        Args:
            noise: 噪声模型对象；不传则默认无噪声。
        """
        self.noise = noise if noise is not None else NoiseModel(enabled=False)
        self._last_run: Optional[dict[str, Any]] = None

    # ---- Circuit builders (tutorial-style convenience wrappers) ----
    def build_encoding_circuit(self, log_qb_idx: int = 0) -> stim.Circuit:
        """构建单个逻辑块的编码电路。"""
        return encoding_circuit(log_qb_idx=log_qb_idx)

    def build_logical_gate(self, gate: str, log_qb_idx: int = 0) -> stim.Circuit:
        """构建逻辑单比特门电路片段。"""
        return logical_single_qubit_gate(gate=gate, log_qb_idx=log_qb_idx)

    def build_state_prep_circuit(self, initial_state: str = "+Z") -> stim.Circuit:
        """构建“编码 + 初态制备”组合电路。"""
        c = stim.Circuit()
        c += encoding_circuit()
        c += prepare_stab_eigenstate(initial_state)
        return c

    def build_single_stabilizer_circuit(self, stabilizer: str, ancilla: int = 8) -> stim.Circuit:
        """按名字构建单稳定子测量子电路（如 S1/S4）。"""
        names = [s["name"] for s in STABILIZER_SEQUENCE]
        if stabilizer not in names:
            raise ValueError(f"Unknown stabilizer {stabilizer}. Choices: {names}")
        stab = next(s for s in STABILIZER_SEQUENCE if s["name"] == stabilizer)
        return measure_single_stabilizer(stab, ancilla=ancilla)

    def build_syndrome_schedule_circuit(
        self,
        n_steps: int = 60,
        ancilla: int = 8,
        detail: Literal["detailed", "coarse"] = "detailed",
    ) -> stim.Circuit:
        """构建固定顺序 `S1..S6` 循环的 syndrome 测量电路。

        Args:
            n_steps: 稳定子测量步数。
            ancilla: syndrome ancilla 索引。
            detail:
                - "detailed": 逐 step 展开全部门操作。
                - "coarse": 尽量按完整 6-step cycle 折叠为 REPEAT 块，便于粗粒度查看。
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        c = stim.Circuit()

        if detail == "detailed":
            for step in range(n_steps):
                stab = STABILIZER_SEQUENCE[step % 6]
                c += measure_single_stabilizer(stab, ancilla=ancilla)
            return c

        if detail != "coarse":
            raise ValueError(f"Unknown detail level: {detail}")

        # Coarse mode: represent full 6-step rounds via REPEAT for compact diagrams.
        full_cycles = n_steps // 6
        remainder_steps = n_steps % 6

        if full_cycles > 0:
            cycle = stim.Circuit()
            for i in range(6):
                cycle += measure_single_stabilizer(STABILIZER_SEQUENCE[i], ancilla=ancilla)
            cycle.append("TICK")
            c.append(stim.CircuitRepeatBlock(full_cycles, cycle))

        for i in range(remainder_steps):
            c += measure_single_stabilizer(STABILIZER_SEQUENCE[i], ancilla=ancilla)

        return c

    def build_full_flow_circuit_for_diagram(
        self,
        initial_state: str = "+Z",
        meas_basis: str = "Z",
        n_steps: int = 12,
        detail: Literal["detailed", "coarse"] = "detailed",
    ) -> stim.Circuit:
        """构建用于画图检查的静态全流程电路。

        该电路主要用于可视化，不包含自适应控制逻辑。
        """
        c = stim.Circuit()
        c += encoding_circuit()
        c += prepare_stab_eigenstate(initial_state)
        c += self.build_syndrome_schedule_circuit(n_steps=n_steps, ancilla=8, detail=detail)
        c += rotate_to_measurement_basis(meas_basis)
        c += measure_logical_qubits()
        return c

    # ---- Diagram helpers ----
    def get_diagram(self, circuit: stim.Circuit, diagram_type: str = "timeline-text") -> str:
        """返回 Stim 电路图字符串（不直接打印）。"""
        return str(circuit.diagram(diagram_type))

    def print_diagram(self, circuit: stim.Circuit, diagram_type: str = "timeline-text") -> None:
        """直接打印 Stim 电路图，便于命令行快速检查。"""
        print(self.get_diagram(circuit, diagram_type=diagram_type))

    def save_diagram(
        self,
        circuit: stim.Circuit,
        output_path: str,
        diagram_type: str = "timeline-text",
    ) -> None:
        """把 Stim 电路图保存到文件。

        Args:
            circuit: 要导出的电路。
            output_path: 输出文件路径（如 `diagram.svg` / `diagram.html` / `diagram.txt`）。
            diagram_type: Stim 的图类型参数（timeline-text/timeline-svg/...）。
        """
        out = Path(output_path)

        if out.suffix.lower() == ".pdf":
            # PDF export path: render SVG first, then convert SVG -> PDF.
            svg_type = diagram_type
            if "svg" not in svg_type:
                svg_type = "timeline-svg"
            content = self.get_diagram(circuit, diagram_type=svg_type)

            # timeline-svg-html contains wrappers; extract the raw <svg> block.
            if svg_type.endswith("html"):
                start = content.find("<svg")
                end = content.rfind("</svg>")
                if start != -1 and end != -1:
                    content = content[start : end + len("</svg>")]

            try:
                import cairosvg
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Saving PDF requires 'cairosvg'. Install it in your env, e.g. "
                    "'conda install -n physics -c conda-forge cairosvg' or "
                    "'pip install cairosvg'."
                ) from exc

            cairosvg.svg2pdf(bytestring=content.encode("utf-8"), write_to=str(out))
            return

        content = self.get_diagram(circuit, diagram_type=diagram_type)
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)

    # ---- Validation helpers ----
    def validate_encoding(self, shots: int = 100) -> dict[str, Any]:
        """做编码阶段的快速 sanity-check。

        检查项：
            - 编码 ancilla 测到 0 的比例
            - 最终重构逻辑 Z_L 为 0 的比例

        Returns:
            包含上述比例与 shot 数的字典。
        """
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
        """验证某个“初态 + 测量基”组合是否给出期望输出。

        Returns:
            含成功率与每个 shot 逻辑测量值的字典。
        """
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
        shot_workers: int = 1,
    ) -> dict[str, Any]:
        """运行完整实验并缓存最近一次结果。

        Returns:
            含参数回显、逐 shot 结果、成功率的汇总字典。
        """
        results = steane_code_exp_sequential(
            initial_state=initial_state,
            meas_basis=meas_basis,
            n_steps=n_steps,
            shots=shots,
            syndrome_mode=syndrome_mode,
            noise=self.noise,
            shot_workers=shot_workers,
        )
        out = {
            "initial_state": initial_state,
            "meas_basis": meas_basis,
            "n_steps": n_steps,
            "shots": shots,
            "shot_workers": int(shot_workers),
            "syndrome_mode": syndrome_mode,
            "results": results,
            "success_rate": float(np.mean(results)) if results else 0.0,
        }
        self._last_run = out
        return out

    def run_experiment_with_trace(
        self,
        initial_state: str = "+Z",
        meas_basis: str = "Z",
        n_steps: int = 60,
        shots: int = 100,
        syndrome_mode: Literal["MV", "DE"] = "MV",
        shot_workers: int = 1,
    ) -> dict[str, Any]:
        """运行完整实验并返回每个 shot 的中间轨迹。"""
        out = steane_code_exp_sequential_with_trace(
            initial_state=initial_state,
            meas_basis=meas_basis,
            n_steps=n_steps,
            shots=shots,
            syndrome_mode=syndrome_mode,
            noise=self.noise,
            shot_workers=shot_workers,
        )
        self._last_run = out
        return out

    def get_results(self) -> dict[str, Any]:
        """获取最近一次 `run_experiment` 的结果。"""
        if self._last_run is None:
            raise RuntimeError("No experiment has been run yet. Call run_experiment first.")
        return self._last_run


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Steane sequential-QEC simulator")
    parser.add_argument("--initial-state", default="+Z", choices=["+Z", "-Z", "+X", "-X", "+Y", "-Y"])
    parser.add_argument("--meas-basis", default="Z", choices=["X", "Y", "Z"])
    parser.add_argument("--n-steps", type=int, default=60, help="Total syndrome-measurement steps.")
    parser.add_argument("--shots", type=int, default=200, help="Number of Monte-Carlo shots.")
    parser.add_argument("--shot-workers", type=int, default=1, help="Parallel workers across shots (summary mode).")
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
        "--diagram-detail",
        default="detailed",
        choices=["detailed", "coarse"],
        help="Diagram granularity for syndrome/full circuits.",
    )
    parser.add_argument(
        "--save-diagram",
        default="",
        help="Save selected diagram to file path. Used with --print-diagram.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run quick tutorial-style validation checks before experiment.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    """脚本入口。

    逻辑：
        1) 解析参数与日志等级
        2) 可选：只打印电路图并退出
        3) 可选：先做快速验证
        4) 运行完整实验并打印摘要
    """
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
            circuit = sim.build_syndrome_schedule_circuit(
                n_steps=args.n_steps,
                detail=args.diagram_detail,
            )
        else:
            circuit = sim.build_full_flow_circuit_for_diagram(
                initial_state=args.initial_state,
                meas_basis=args.meas_basis,
                n_steps=args.n_steps,
                detail=args.diagram_detail,
            )
        if args.save_diagram:
            sim.save_diagram(
                circuit=circuit,
                output_path=args.save_diagram,
                diagram_type=args.diagram_type,
            )
            print(f"Diagram saved to: {args.save_diagram}")
        else:
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
        shot_workers=args.shot_workers,
    )

    print("Steane sequential-QEC summary")
    print(f"  initial_state: {args.initial_state}")
    print(f"  meas_basis:    {args.meas_basis}")
    print(f"  n_steps:       {args.n_steps}")
    print(f"  shots:         {args.shots}")
    print(f"  shot_workers:  {args.shot_workers}")
    print(f"  mode:          {args.mode}")
    print(f"  success_rate:  {summary['success_rate']:.4f}")


if __name__ == "__main__":
    main()
