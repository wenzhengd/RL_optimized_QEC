"""Expr4 V2 cycle sweep: full-channel RL vs Expr1 transfer vs fixed_zero."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .eval_steane_cycle_sweep import (
    _augment_metrics,
    _evaluate_fixed_baselines,
    _evaluate_single_policy,
    _parse_round_sweep,
    _run_cycle_sweep,
    _train_policy,
)
from .eval_steane_ppo import build_arg_parser as build_benchmark_arg_parser
from .eval_steane_ppo import load_policy_checkpoint, save_policy_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_benchmark_arg_parser()
    parser.description = "Expr4 V2: train one full-channel policy and compare it against Expr1 transfer and fixed_zero over a cycle sweep."
    parser.add_argument(
        "--cycle-sweep-rounds",
        type=str,
        default="5,10,15,20,25,30,35,40,45,50",
        help="Comma-separated evaluation-time n_rounds sweep.",
    )
    parser.add_argument(
        "--transfer-source-checkpoint",
        type=str,
        default="",
        help="Optional Expr1 checkpoint to evaluate as transferred policy on the same full-composite test channel.",
    )
    parser.add_argument(
        "--primary-policy-label",
        type=str,
        default="full_channel_RL",
    )
    parser.add_argument(
        "--transfer-policy-label",
        type=str,
        default="full_channel_transfer_expr1",
    )
    parser.add_argument(
        "--save-primary-policy-checkpoint",
        type=str,
        default="",
        help="Optional path to save the trained full-channel showcase policy checkpoint.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _run_transfer_cycle_sweep(
    checkpoint_path: Path,
    *,
    primary_steane_cfg,
    args: argparse.Namespace,
    cycle_rounds: Sequence[int],
) -> dict[str, Any]:
    model, ppo_cfg, payload = load_policy_checkpoint(checkpoint_path, device=str(args.device))
    rows = _run_cycle_sweep(model, primary_steane_cfg, args, cycle_rounds)
    return {
        "label": str(args.transfer_policy_label),
        "source_checkpoint": str(checkpoint_path),
        "source_args": payload.get("args", {}),
        "source_ppo_cfg": payload.get("ppo_cfg", {}),
        "ppo_cfg": asdict(ppo_cfg),
        "cycle_sweep": rows,
    }


def run_cycle_sweep_v2(args: argparse.Namespace) -> dict[str, Any]:
    cycle_rounds = _parse_round_sweep(args.cycle_sweep_rounds)

    primary_model, primary_history, primary_steane_cfg, primary_ppo_cfg = _train_policy(args)
    if args.save_primary_policy_checkpoint:
        save_policy_checkpoint(
            checkpoint_path=Path(args.save_primary_policy_checkpoint),
            model=primary_model,
            args=args,
            ppo_cfg=primary_ppo_cfg,
            steane_cfg=primary_steane_cfg,
        )

    primary_rows = _run_cycle_sweep(primary_model, primary_steane_cfg, args, cycle_rounds)
    transfer_report = None
    if args.transfer_source_checkpoint:
        transfer_report = _run_transfer_cycle_sweep(
            checkpoint_path=Path(args.transfer_source_checkpoint),
            primary_steane_cfg=primary_steane_cfg,
            args=args,
            cycle_rounds=cycle_rounds,
        )

    fixed_rows = []
    for row in primary_rows:
        fixed_rows.append(
            {
                "n_rounds": int(row["n_rounds"]),
                "n_steps": int(row["n_steps"]),
                "nominal_circuit_timing_per_rl_step": row["nominal_circuit_timing_per_rl_step"],
                "fixed_zero": row["fixed_zero"],
            }
        )

    report: dict[str, Any] = {
        "cycle_rounds": [int(v) for v in cycle_rounds],
        "test_condition_args": vars(args),
        "primary_policy": {
            "label": str(args.primary_policy_label),
            "train_args": vars(args),
            "steane_cfg": asdict(primary_steane_cfg),
            "ppo_cfg": asdict(primary_ppo_cfg),
            "final_mean_rollout_reward": float(primary_history["mean_reward_rollout"][-1]),
            "cycle_sweep": primary_rows,
        },
        "transfer_policy": transfer_report,
        "fixed_zero_policy": {
            "label": "fixed_zero",
            "cycle_sweep": fixed_rows,
        },
    }
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def print_report(report: dict[str, Any]) -> None:
    print("Expr4 V2 cycle sweep finished.")
    print(f"Primary policy: {report['primary_policy']['label']}")
    if report["transfer_policy"] is not None:
        print(f"Transfer policy: {report['transfer_policy']['label']}")
    print("n_rounds | primary_success | fixed_zero_success | transfer_success")
    transfer_rows = None
    if report["transfer_policy"] is not None:
        transfer_rows = {int(r["n_rounds"]): r for r in report["transfer_policy"]["cycle_sweep"]}
    fixed_rows = {int(r["n_rounds"]): r for r in report["fixed_zero_policy"]["cycle_sweep"]}
    for row in report["primary_policy"]["cycle_sweep"]:
        rounds = int(row["n_rounds"])
        fixed = fixed_rows[rounds]["fixed_zero"]["success_rate"]
        transfer = float("nan")
        if transfer_rows is not None:
            transfer = float(transfer_rows[rounds]["learned"]["success_rate"])
        print(
            f"{rounds:>8d} | "
            f"{row['learned']['success_rate']:.6f} | "
            f"{fixed:.6f} | "
            f"{transfer:.6f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_cycle_sweep_v2(args)
    print_report(report)


if __name__ == "__main__":
    main()
