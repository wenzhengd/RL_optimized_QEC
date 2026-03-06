"""Noise-channel factory utilities for simulator and RL workflows.

This module centralizes channel selection so scripts do not hardcode
`noise_engine` classes directly. It keeps existing behavior by default
while making it easy to add new channel families.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence, Tuple

import numpy as np

try:
    from .noise_engine import (
        GateDurations,
        GoogleLikeDepolarizingNoiseModel,
        GoogleLikeGateSpecificNoiseModel,
        NoiseModel,
        TimeDependentPauliNoiseModel,
    )
except ImportError:
    # Script-local import path, e.g. from `quantum_simulation/` directory.
    from noise_engine import (  # type: ignore
        GateDurations,
        GoogleLikeDepolarizingNoiseModel,
        GoogleLikeGateSpecificNoiseModel,
        NoiseModel,
        TimeDependentPauliNoiseModel,
    )

SteaneNoiseChannel = Literal[
    "auto",
    "google_global",
    "google_gate_specific",
    "idle_depolarizing",
    "parametric_google",
    "correlated_pauli_noise_channel",
]


def available_steane_noise_channels() -> Tuple[str, ...]:
    """Return supported channel keys for Steane RL/sweep paths."""
    return (
        "auto",
        "google_global",
        "google_gate_specific",
        "idle_depolarizing",
        "parametric_google",
        "correlated_pauli_noise_channel",
    )


def _resolve_auto_channel(channel: str, control_mode: str) -> str:
    """Map legacy control-mode flow to explicit channel key."""
    if channel != "auto":
        return channel
    if control_mode == "global":
        return "google_global"
    if control_mode == "gate_specific":
        return "google_gate_specific"
    raise ValueError(f"Unknown control_mode for auto channel resolution: {control_mode}")


def build_idle_depolarizing_noise_model(
    p_total_per_idle: float,
    *,
    idle_ns: float | None = None,
    axis_weights: Sequence[float] = (1.0, 1.0, 1.0),
    enabled: bool = True,
) -> TimeDependentPauliNoiseModel:
    """Build a time-dependent Pauli model equivalent to idle depolarizing sweep.

    Args:
        p_total_per_idle: Total depolarizing probability per idle window.
        idle_ns: Idle duration in ns. Defaults to `GateDurations().idle_ns`.
        axis_weights: Relative X/Y/Z split weights.
        enabled: Forwarded to `TimeDependentPauliNoiseModel`.
    """
    p_total = max(0.0, float(p_total_per_idle))
    idle = float(idle_ns) if idle_ns is not None else float(GateDurations().idle_ns)
    if idle <= 0.0:
        raise ValueError("idle_ns must be positive.")

    w = np.asarray(axis_weights, dtype=float).reshape(-1)
    if w.shape[0] != 3:
        raise ValueError("axis_weights must contain exactly 3 values.")
    if np.any(w < 0.0):
        raise ValueError("axis_weights must be non-negative.")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("axis_weights sum must be positive.")
    w = w / w_sum

    # per-axis probability over one idle, then converted to per-ns rate.
    px = float(w[0]) * p_total
    py = float(w[1]) * p_total
    pz = float(w[2]) * p_total
    return TimeDependentPauliNoiseModel(
        p_x=px / idle,
        p_y=py / idle,
        p_z=pz / idle,
        enabled=enabled,
    )


def correlated_pauli_model_kernel(
    *,
    t_ns: float,
    qubit_index: int,
    idle_ns: float,
    action_vec: np.ndarray,
    optimal_vec: np.ndarray,
    p_1q_base: float,
    sensitivity_1q: float,
    p_clip_max: float,
    regime_a: float,
    regime_b: float,
) -> float:
    """User-editable physics model hook for correlated Pauli channel.

    This is the single place you should replace with your own physical model.
    Contract:
      - Return total Pauli probability `p_total(q,t)` for one idle window.
      - Must satisfy `0 <= p_total <= p_clip_max`.
      - `regime_a`, `regime_b` are free control parameters for regime sweeps.
    """
    q = int(qubit_index)
    idle = float(idle_ns)
    if idle <= 0.0:
        return 0.0

    delta_sq = np.square(action_vec - optimal_vec)
    mismatch_global = float(np.mean(delta_sq))
    p_total_nominal = float(np.clip((float(p_1q_base) + float(sensitivity_1q) * mismatch_global) * float(regime_a), 0.0, float(p_clip_max)))

    # Default prototype correlation model (replace freely).
    corr = float(np.clip(np.tanh(float(regime_b)), 0.0, 0.99))
    idx = q % action_vec.size
    local_mismatch = float(delta_sq[idx])
    local_scale = 1.0 + 0.5 * (local_mismatch - mismatch_global) / (mismatch_global + 1e-12)
    local_scale = float(np.clip(local_scale, 0.25, 2.0))
    p_local = p_total_nominal * local_scale

    amp = 0.35
    control_phase = float(np.sum(action_vec))
    x = float(t_ns) / idle
    global_wave = float(np.sin(2.0 * np.pi * x / 48.0 + 0.1 * control_phase))
    local_wave = float(np.sin(2.0 * np.pi * x / 9.0 + 0.37 * q + 0.03 * control_phase))
    modulation = corr * global_wave + (1.0 - corr) * local_wave
    return float(np.clip(p_local * (1.0 + amp * modulation), 0.0, float(p_clip_max)))


def build_correlated_pauli_noise_channel(
    *,
    action: np.ndarray,
    optimal_control_fn: Callable[[float], Sequence[float]],
    p_1q_base: float,
    sensitivity_1q: float,
    p_clip_max: float,
    regime_a: float,
    regime_b: float,
    axis_weights: Sequence[float] = (1.0, 1.0, 1.0),
    idle_ns: float | None = None,
    enabled: bool = True,
) -> tuple[TimeDependentPauliNoiseModel, float]:
    """Build a custom correlated Pauli idle-noise channel.

    Design intent:
      - Beyond standard idle depolarizing and beyond Google gate-depolarizing mapping.
      - Uses `correlated_pauli_model_kernel(...)` as the only physics hook.

    Returns:
      `(noise_model, p_total_proxy)`, where `p_total_proxy` is the nominal
      per-idle total Pauli probability scale used for logging.
    """
    action_vec = np.asarray(action, dtype=float).reshape(-1)
    if action_vec.size == 0:
        raise ValueError("action must be non-empty for correlated_pauli_noise_channel.")

    opt = np.asarray(optimal_control_fn(0.0), dtype=float).reshape(-1)
    if opt.shape != action_vec.shape:
        raise ValueError(
            "optimal_control_fn output shape mismatch for correlated channel: "
            f"action={action_vec.shape}, optimal={opt.shape}"
        )

    idle = float(idle_ns) if idle_ns is not None else float(GateDurations().idle_ns)
    if idle <= 0.0:
        raise ValueError("idle_ns must be positive.")

    w = np.asarray(axis_weights, dtype=float).reshape(-1)
    if w.shape[0] != 3:
        raise ValueError("axis_weights must contain exactly 3 values.")
    if np.any(w < 0.0):
        raise ValueError("axis_weights must be non-negative.")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("axis_weights sum must be positive.")
    w = w / w_sum

    reg_a = max(0.0, float(regime_a))
    reg_b = max(0.0, float(regime_b))
    clip_max = float(p_clip_max)

    delta_sq = np.square(action_vec - opt)
    mismatch_global = float(np.mean(delta_sq))
    p_total_nominal = float(np.clip((float(p_1q_base) + float(sensitivity_1q) * mismatch_global) * reg_a, 0.0, clip_max))

    def _p_total_tq(t_ns: float, qubit_index: int) -> float:
        return correlated_pauli_model_kernel(
            t_ns=float(t_ns),
            qubit_index=int(qubit_index),
            idle_ns=idle,
            action_vec=action_vec,
            optimal_vec=opt,
            p_1q_base=float(p_1q_base),
            sensitivity_1q=float(sensitivity_1q),
            p_clip_max=float(p_clip_max),
            regime_a=reg_a,
            regime_b=reg_b,
        )

    def _p_axis(t_ns: float, qubit_index: int, weight: float) -> float:
        return (_p_total_tq(t_ns, qubit_index) * float(weight)) / idle

    noise = TimeDependentPauliNoiseModel(
        p_x=lambda t, q: _p_axis(t, q, float(w[0])),
        p_y=lambda t, q: _p_axis(t, q, float(w[1])),
        p_z=lambda t, q: _p_axis(t, q, float(w[2])),
        enabled=enabled,
    )
    return noise, p_total_nominal


def build_steane_rl_noise_model(
    *,
    noise_channel: str,
    control_mode: str,
    action: np.ndarray,
    optimal_control_fn: Callable[[float], Sequence[float]],
    p_1q_base: float,
    p_2q_base: float,
    sensitivity_1q: float,
    sensitivity_2q: float,
    n_1q_slots: int,
    n_2q_slots: int,
    p_clip_max: float,
    idle_p_total_per_idle: float,
    idle_px_weight: float,
    idle_py_weight: float,
    idle_pz_weight: float,
    channel_regime_a: float,
    channel_regime_b: float,
    enabled: bool = True,
) -> tuple[NoiseModel, float, float, str]:
    """Construct the noise model used by Steane RL adapter.

    Returns:
        `(noise_model, p_1q_proxy, p_2q_proxy, resolved_channel_name)`.
    """
    resolved = _resolve_auto_channel(str(noise_channel), str(control_mode))

    # Regime knobs are channel-level parameters intended for future sweeps.
    reg_a = max(0.0, float(channel_regime_a))
    reg_b = max(0.0, float(channel_regime_b))

    if resolved == "google_global":
        noise = GoogleLikeDepolarizingNoiseModel(
            control=action,
            optimal_control_fn=optimal_control_fn,
            p_1q_base=float(p_1q_base) * reg_a,
            p_2q_base=float(p_2q_base) * reg_b,
            sensitivity_1q=float(sensitivity_1q) * reg_a,
            sensitivity_2q=float(sensitivity_2q) * reg_b,
            p_clip_max=float(p_clip_max),
            enabled=enabled,
        )
        p_1q, p_2q = noise.effective_error_rates(0.0)
        return noise, float(p_1q), float(p_2q), resolved

    if resolved == "google_gate_specific":
        noise = GoogleLikeGateSpecificNoiseModel(
            control=action,
            optimal_control_fn=optimal_control_fn,
            p_1q_base=float(p_1q_base) * reg_a,
            p_2q_base=float(p_2q_base) * reg_b,
            sensitivity_1q=float(sensitivity_1q) * reg_a,
            sensitivity_2q=float(sensitivity_2q) * reg_b,
            n_1q_slots=int(n_1q_slots),
            n_2q_slots=int(n_2q_slots),
            p_clip_max=float(p_clip_max),
            enabled=enabled,
        )
        p_1q, p_2q = noise.effective_error_rates(0.0)
        return noise, float(p_1q), float(p_2q), resolved

    if resolved == "idle_depolarizing":
        total_idle = max(0.0, float(idle_p_total_per_idle))
        weights = (float(idle_px_weight), float(idle_py_weight), float(idle_pz_weight))
        noise = build_idle_depolarizing_noise_model(
            p_total_per_idle=total_idle,
            axis_weights=weights,
            enabled=enabled,
        )
        # Proxy rates for logging consistency in downstream code.
        return noise, float(total_idle), float(total_idle), resolved

    if resolved == "parametric_google":
        # Same functional form as gate-specific Google-like channel, but driven by
        # explicit regime knobs (a,b) to support future regime sweeps.
        noise = GoogleLikeGateSpecificNoiseModel(
            control=action,
            optimal_control_fn=optimal_control_fn,
            p_1q_base=float(p_1q_base) * reg_a,
            p_2q_base=float(p_2q_base) * reg_b,
            sensitivity_1q=float(sensitivity_1q) * reg_a,
            sensitivity_2q=float(sensitivity_2q) * reg_b,
            n_1q_slots=int(n_1q_slots),
            n_2q_slots=int(n_2q_slots),
            p_clip_max=float(p_clip_max),
            enabled=enabled,
        )
        p_1q, p_2q = noise.effective_error_rates(0.0)
        return noise, float(p_1q), float(p_2q), resolved

    if resolved == "correlated_pauli_noise_channel":
        noise, p_proxy = build_correlated_pauli_noise_channel(
            action=action.astype(float),
            optimal_control_fn=optimal_control_fn,
            p_1q_base=float(p_1q_base),
            sensitivity_1q=float(sensitivity_1q),
            p_clip_max=float(p_clip_max),
            regime_a=reg_a,
            regime_b=reg_b,
            axis_weights=(float(idle_px_weight), float(idle_py_weight), float(idle_pz_weight)),
            enabled=enabled,
        )
        # This channel is idle-Pauli based; report one proxy scale for both slots.
        return noise, float(p_proxy), float(p_proxy), resolved

    raise ValueError(
        f"Unknown noise_channel '{noise_channel}'. "
        f"Supported: {', '.join(available_steane_noise_channels())}"
    )
