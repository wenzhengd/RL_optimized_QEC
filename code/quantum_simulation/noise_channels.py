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
        HiddenMarkovCorrelatedPauliNoiseModel,
        NoiseModel,
        TimeDependentPauliNoiseModel,
    )
except ImportError:
    # Script-local import path, e.g. from `quantum_simulation/` directory.
    from noise_engine import (  # type: ignore
        GateDurations,
        GoogleLikeDepolarizingNoiseModel,
        GoogleLikeGateSpecificNoiseModel,
        HiddenMarkovCorrelatedPauliNoiseModel,
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
CorrelatedStrengthMode = Literal["per_window", "per_circuit"]


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


def build_correlated_pauli_noise_channel(
    *,
    action: np.ndarray,
    optimal_control_fn: Callable[[float], Sequence[float]],
    p_1q_base: float,
    sensitivity_1q: float,
    p_clip_max: float,
    corr_strength_g: float,
    corr_frequency_hz: float,
    corr_strength_mode: CorrelatedStrengthMode = "per_window",
    corr_normalization_windows: int | None = None,
    axis_weights: Sequence[float] = (1.0, 1.0, 1.0),
    enabled: bool = True,
) -> tuple[HiddenMarkovCorrelatedPauliNoiseModel, float]:
    """Build a direction-independent Hidden-Markov Pauli idle-noise channel.

    This construction is:
      - qubit independent: each qubit has independent hidden chains
      - direction independent: X/Y/Z have independent hidden chains with
        identical (f, g) parameters
      - temporally correlated: each chain uses a two-state Markov telegraph

    Returns:
      `(noise_model, p_total_effective)` where `p_total_effective` is the
      stationary mean total non-identity Pauli probability per idle window.
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

    w = np.asarray(axis_weights, dtype=float).reshape(-1)
    if w.shape[0] != 3:
        raise ValueError("axis_weights must contain exactly 3 values.")
    if np.any(w < 0.0):
        raise ValueError("axis_weights must be non-negative.")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("axis_weights sum must be positive.")
    w = w / w_sum

    # This channel enforces direction-independence.
    if not np.allclose(w, np.full(3, 1.0 / 3.0), atol=1e-12):
        raise ValueError(
            "correlated_pauli_noise_channel enforces direction-independent noise; "
            "set idle_px_weight=idle_py_weight=idle_pz_weight."
        )

    g = max(0.0, float(corr_strength_g))
    strength_mode = str(corr_strength_mode)
    if strength_mode not in ("per_window", "per_circuit"):
        raise ValueError(
            f"Unknown corr_strength_mode={corr_strength_mode}. "
            "Expected one of: per_window, per_circuit."
        )
    f_hz = max(0.0, float(corr_frequency_hz))
    clip_max = float(p_clip_max)

    # Mean total non-identity probability calibration.
    delta_sq = np.square(action_vec - opt)
    mismatch_global = float(np.mean(delta_sq))
    base_strength = float(p_1q_base) + float(sensitivity_1q) * mismatch_global
    p_total_target = float(np.clip(base_strength * g, 0.0, clip_max))
    norm_windows = 1
    if strength_mode == "per_window":
        p_total_nominal = p_total_target
    else:
        norm_windows = int(corr_normalization_windows) if corr_normalization_windows is not None else 1
        if norm_windows <= 0:
            raise ValueError(
                "corr_normalization_windows must be a positive integer when "
                "corr_strength_mode='per_circuit'."
            )
        # Convert one-circuit budget to one-idle-window mean probability.
        # If each idle window had mean p and windows were independent,
        #   P(any non-identity over N windows) = 1 - (1-p)^N.
        # We invert this mapping so `g` can be interpreted at circuit scale.
        p_total_nominal = float(1.0 - float((1.0 - p_total_target) ** (1.0 / float(norm_windows))))

    p_total_nominal = float(
        np.clip(
            p_total_nominal,
            0.0,
            clip_max,
        )
    )

    # For independent X/Y/Z Bernoulli flips with per-axis mean p_axis:
    #   p_total = 1 - P(I) = 1 - ((1-p)^3 + p^3) = 3p(1-p)
    # so p_axis = (1 - sqrt(1 - 4*p_total/3)) / 2.
    p_total_effective = float(np.clip(p_total_nominal, 0.0, 0.75 - 1e-12))
    disc = float(max(0.0, 1.0 - (4.0 / 3.0) * p_total_effective))
    p_axis_mean = 0.5 * (1.0 - float(np.sqrt(disc)))

    # Two-state on/off telegraph with stationary mean p_axis_mean.
    p_low = 0.0
    p_high = float(np.clip(2.0 * p_axis_mean, 0.0, 1.0))

    idle_s = float(GateDurations().idle_ns) * 1e-9
    if idle_s <= 0.0:
        raise ValueError("idle_ns must be positive.")
    rho = float(np.exp(-f_hz * idle_s))
    gamma = float(np.clip((1.0 - rho) / 2.0, 0.0, 0.499999))

    transition = np.array([[1.0 - gamma, gamma], [gamma, 1.0 - gamma]], dtype=float)
    # Seed from global numpy RNG to preserve reproducibility when global seeding is used.
    seed = int(np.random.randint(0, 2**32 - 1))
    noise = HiddenMarkovCorrelatedPauliNoiseModel(
        p_by_state=[p_low, p_high],
        transition_matrix=transition,
        initial_distribution=[0.5, 0.5],
        durations=GateDurations(),
        enabled=enabled,
        random_seed=seed,
    )
    noise.model_metadata = {
        "corr_f_hz": float(f_hz),
        "corr_g": float(g),
        "corr_g_mode_per_circuit": int(strength_mode == "per_circuit"),
        "corr_norm_windows": int(norm_windows),
        "gamma": float(gamma),
        "p_total_target": float(p_total_target),
        "p_total_nominal": float(p_total_nominal),
        "p_total_effective": float(p_total_effective),
        "p_axis_mean": float(p_axis_mean),
        "p_axis_low": float(p_low),
        "p_axis_high": float(p_high),
        "direction_independent": 1,
        "qubit_independent": 1,
    }
    return noise, p_total_effective


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
    channel_corr_f: float,
    channel_corr_g: float,
    channel_corr_g_mode: CorrelatedStrengthMode,
    channel_corr_windows_per_step: int,
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
            corr_strength_g=float(channel_corr_g),
            corr_strength_mode=channel_corr_g_mode,
            corr_normalization_windows=int(channel_corr_windows_per_step),
            corr_frequency_hz=float(channel_corr_f),
            axis_weights=(float(idle_px_weight), float(idle_py_weight), float(idle_pz_weight)),
            enabled=enabled,
        )
        # This channel is idle-Pauli based; report one proxy scale for both slots.
        return noise, float(p_proxy), float(p_proxy), resolved

    raise ValueError(
        f"Unknown noise_channel '{noise_channel}'. "
        f"Supported: {', '.join(available_steane_noise_channels())}"
    )
