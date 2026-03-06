"""Steane code-family builder for rl_train."""

from __future__ import annotations

import argparse

from ..env import ExternalSimulatorEnv
from ..interfaces import RewardFn
from ..steane_adapter import (
    SteaneAdapterConfig,
    SteaneOnlineSteeringSimulator,
    clipped_identity_action_mapper,
)
from .base import CodeComponents


def build_steane_components(args: argparse.Namespace, reward_fn: RewardFn) -> CodeComponents:
    """Construct Steane config/simulator/env bundle from argparse args."""
    steane_cfg = SteaneAdapterConfig(
        n_rounds=args.steane_n_rounds,
        shots_per_step=args.steane_shots_per_step,
        control_mode=args.steane_control_mode,
        control_dim=args.steane_control_dim,
        n_1q_control_slots=args.steane_n_1q_control_slots,
        n_2q_control_slots=args.steane_n_2q_control_slots,
        syndrome_mode=args.steane_syndrome_mode,
        stepping_mode=args.steane_stepping_mode,
        drift_period_steps=args.steane_drift_period_steps,
        drift_amplitude=args.steane_drift_amplitude,
        p_1q_base=args.steane_p1q_base,
        p_2q_base=args.steane_p2q_base,
        sensitivity_1q=args.steane_sensitivity_1q,
        sensitivity_2q=args.steane_sensitivity_2q,
        p_clip_max=args.steane_p_clip_max,
        noise_channel=args.steane_noise_channel,
        idle_p_total_per_idle=args.steane_idle_p_total_per_idle,
        idle_px_weight=args.steane_idle_px_weight,
        idle_py_weight=args.steane_idle_py_weight,
        idle_pz_weight=args.steane_idle_pz_weight,
        channel_corr_f=args.steane_channel_corr_f,
        channel_corr_g=args.steane_channel_corr_g,
        channel_regime_a=args.steane_channel_regime_a,
        channel_regime_b=args.steane_channel_regime_b,
        shot_workers=args.steane_shot_workers,
        collect_traces=args.steane_collect_traces,
        reset_drift_on_episode=args.steane_reset_drift_on_episode,
        expose_oracle_metrics=args.steane_expose_oracle_metrics,
        seed=args.seed,
    )
    simulator = SteaneOnlineSteeringSimulator(steane_cfg)
    action_mapper = lambda x: clipped_identity_action_mapper(x, action_limit=args.action_limit)
    env = ExternalSimulatorEnv(
        simulator=simulator,
        max_steps=args.max_steps,
        reward_fn=reward_fn,
        action_mapper=action_mapper,
    )
    return CodeComponents(
        code_family="steane",
        code_cfg=steane_cfg,
        simulator=simulator,
        env=env,
        action_mapper=action_mapper,
        reward_fn=reward_fn,
    )
