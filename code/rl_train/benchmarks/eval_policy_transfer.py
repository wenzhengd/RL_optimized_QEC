"""Evaluate a saved PPO policy checkpoint on a different Steane benchmark condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .eval_steane_ppo import (
    _build_ppo_config,
    _build_steane_components,
    _evaluate_policies,
    _rel_improvement,
    load_policy_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved PPO policy on a transfer target condition.")
    parser.add_argument("--source-checkpoint", type=str, required=True, help="Checkpoint saved by eval_steane_ppo.")
    parser.add_argument(
        "--target-run-json",
        type=str,
        required=True,
        help="Existing benchmark run JSON whose args define the transfer target condition.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save transfer evaluation report.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="If >0, override target run's post_eval_episodes during transfer evaluation.",
    )
    parser.add_argument(
        "--eval-steane-shots-per-step",
        type=int,
        default=0,
        help="If >0, override target run's eval shots per step during transfer evaluation.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def run_transfer(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_checkpoint)
    target_path = Path(args.target_run_json)

    source_model, source_ppo_cfg, source_payload = load_policy_checkpoint(source_path, device=str(args.device))
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    target_args = argparse.Namespace(**target_payload["args"])
    target_args.device = str(args.device)
    target_args.save_json = ""
    if hasattr(target_args, "save_policy_checkpoint"):
        target_args.save_policy_checkpoint = ""
    if int(args.episodes) > 0:
        target_args.post_eval_episodes = int(args.episodes)
    if int(args.eval_steane_shots_per_step) > 0:
        target_args.eval_steane_shots_per_step = int(args.eval_steane_shots_per_step)

    target_steane_cfg, target_simulator, _ = _build_steane_components(target_args)
    target_ppo_cfg = _build_ppo_config(target_args, target_simulator)

    if int(source_ppo_cfg.obs_dim) != int(target_ppo_cfg.obs_dim):
        raise ValueError(
            f"Obs dim mismatch: source={source_ppo_cfg.obs_dim}, target={target_ppo_cfg.obs_dim}."
        )
    if int(source_ppo_cfg.theta_dim) != int(target_ppo_cfg.theta_dim):
        raise ValueError(
            f"Action dim mismatch: source={source_ppo_cfg.theta_dim}, target={target_ppo_cfg.theta_dim}."
        )

    transfer_eval = _evaluate_policies(
        model=source_model,
        cfg=target_ppo_cfg,
        steane_cfg=target_steane_cfg,
        action_limit=float(target_args.action_limit),
        episodes=int(target_args.post_eval_episodes),
        eval_shots_per_step=int(target_args.eval_steane_shots_per_step),
    )

    transfer_learned = transfer_eval["learned"]
    transfer_fixed = transfer_eval["fixed_zero"]
    reference_target = target_payload.get("eval_metrics", {}).get("learned")

    report: dict[str, Any] = {
        "source_checkpoint": str(source_path),
        "target_run_json": str(target_path),
        "source_args": source_payload.get("args", {}),
        "source_ppo_cfg": source_payload.get("ppo_cfg", {}),
        "target_args": target_payload.get("args", {}),
        "transfer_eval_metrics": transfer_eval,
        "transfer_improvement_vs_fixed_zero": {
            "detector_rate": _rel_improvement(transfer_fixed["detector_rate"], transfer_learned["detector_rate"]),
            "ler_proxy": _rel_improvement(transfer_fixed["ler_proxy"], transfer_learned["ler_proxy"]),
        },
        "reference_target_learned_metrics": reference_target,
    }

    if reference_target is not None:
        report["transfer_vs_target_learned"] = {
            "delta_success_rate": float(transfer_learned["success_rate"]) - float(reference_target["success_rate"]),
            "delta_ler_proxy": float(transfer_learned["ler_proxy"]) - float(reference_target["ler_proxy"]),
            "delta_detector_rate": float(transfer_learned["detector_rate"]) - float(reference_target["detector_rate"]),
        }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return report


def main() -> None:
    args = parse_args()
    report = run_transfer(args)
    transfer = report["transfer_eval_metrics"]["learned"]
    print(
        "Transfer learned metrics: "
        f"DR={transfer['detector_rate']:.6f}, "
        f"success={transfer['success_rate']:.6f}, "
        f"LER~={transfer['ler_proxy']:.6f}"
    )
    print(
        "Transfer improvement vs fixed-zero: "
        f"DR={100.0 * report['transfer_improvement_vs_fixed_zero']['detector_rate']:+.2f}%, "
        f"LER~={100.0 * report['transfer_improvement_vs_fixed_zero']['ler_proxy']:+.2f}%"
    )
    if "transfer_vs_target_learned" in report:
        delta = report["transfer_vs_target_learned"]
        print(
            "Transfer vs target-learned: "
            f"delta_success={delta['delta_success_rate']:+.6f}, "
            f"delta_LER~={delta['delta_ler_proxy']:+.6f}"
        )
    if args.output_json:
        print(f"Saved transfer report: {args.output_json}")


if __name__ == "__main__":
    main()
