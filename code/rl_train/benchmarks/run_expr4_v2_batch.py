"""Launch Expr4 V2 cycle-sweep runs as independent per-seed commands."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Expr4 V2 cycle-sweep seeds as independent commands.")
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to launch the benchmark module.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1800,1801,1802",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of concurrent per-seed processes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2",
        help="Directory containing per-seed JSON reports and logs.",
    )
    parser.add_argument(
        "--phase-label",
        type=str,
        default="phaseA_showcase",
        help="Filename prefix for outputs.",
    )
    parser.add_argument(
        "--transfer-source-checkpoint",
        type=str,
        default="code/data_generated/rl_steane_tune_experiments_V2/policy_transfer/expr1_source_seed720_checkpoint.pt",
        help="Expr1 checkpoint injected as transfer policy.",
    )
    parser.add_argument(
        "--steane-noise-channel",
        type=str,
        default="composed_google_gate_specific_correlated",
        help="Noise channel used for the full-composite test condition.",
    )
    parser.add_argument(
        "--regime-a",
        type=float,
        default=0.025,
        help="Full-channel regime_a used for training and testing.",
    )
    parser.add_argument(
        "--regime-b",
        type=float,
        default=0.25,
        help="Full-channel regime_b used for training and testing.",
    )
    parser.add_argument(
        "--corr-f",
        type=float,
        default=10000.0,
        help="Correlated noise frequency parameter.",
    )
    parser.add_argument(
        "--corr-g",
        type=float,
        default=1.0,
        help="Correlated noise strength parameter.",
    )
    parser.add_argument(
        "--corr-g-mode",
        type=str,
        default="per_circuit",
        help="Correlated g scheduling mode.",
    )
    parser.add_argument(
        "--p-meas",
        type=float,
        default=0.003,
        help="Measurement bit-flip probability for the full channel.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=8192,
        help="Primary full-channel RL training budget.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=64,
        help="Primary PPO rollout length.",
    )
    parser.add_argument(
        "--ppo-learning-rate",
        type=float,
        default=1e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--ppo-ent-coef",
        type=float,
        default=1e-3,
        help="PPO entropy coefficient.",
    )
    parser.add_argument(
        "--action-penalty-coef",
        type=float,
        default=0.005,
        help="Steane action penalty coefficient.",
    )
    parser.add_argument(
        "--miscal-penalty-coef",
        type=float,
        default=0.001,
        help="Steane miscalibration penalty coefficient.",
    )
    parser.add_argument(
        "--train-n-rounds",
        type=int,
        default=6,
        help="Training-time Steane n_rounds.",
    )
    parser.add_argument(
        "--train-shots-per-step",
        type=int,
        default=16,
        help="Training-time shots per step.",
    )
    parser.add_argument(
        "--trace-finetune-timesteps",
        type=int,
        default=1024,
        help="Trace finetune timesteps for the primary policy.",
    )
    parser.add_argument(
        "--trace-finetune-rollout-steps",
        type=int,
        default=32,
        help="Trace finetune rollout steps.",
    )
    parser.add_argument(
        "--trace-finetune-shots-per-step",
        type=int,
        default=16,
        help="Trace finetune shots per step.",
    )
    parser.add_argument(
        "--trace-finetune-n-rounds",
        type=int,
        default=6,
        help="Trace finetune n_rounds.",
    )
    parser.add_argument(
        "--post-eval-episodes",
        type=int,
        default=48,
        help="Episodes per evaluated cycle point.",
    )
    parser.add_argument(
        "--eval-shots-per-step",
        type=int,
        default=64,
        help="Evaluation-time shots per step.",
    )
    parser.add_argument(
        "--cycle-sweep-rounds",
        type=str,
        default="5,10,15,20,25,30,35,40,45,50",
        help="Evaluation-time cycle sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands without launching them.",
    )
    return parser


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def build_command(args: argparse.Namespace, seed: int) -> list[str]:
    output_dir = Path(args.output_dir)
    json_path = output_dir / f"{args.phase_label}_seed{seed}.json"
    ckpt_path = output_dir / f"{args.phase_label}_seed{seed}_primary_checkpoint.pt"
    return [
        str(args.python_exe),
        "-m",
        "rl_train.benchmarks.eval_steane_cycle_sweep_v2",
        "--code-family",
        "steane",
        "--device",
        "cpu",
        "--seed",
        str(seed),
        "--steane-noise-channel",
        str(args.steane_noise_channel),
        "--steane-channel-regime-a",
        str(args.regime_a),
        "--steane-channel-regime-b",
        str(args.regime_b),
        "--steane-channel-corr-f",
        str(args.corr_f),
        "--steane-channel-corr-g",
        str(args.corr_g),
        "--steane-channel-corr-g-mode",
        str(args.corr_g_mode),
        "--steane-measurement-bitflip-prob",
        str(args.p_meas),
        "--total-timesteps",
        str(args.total_timesteps),
        "--rollout-steps",
        str(args.rollout_steps),
        "--ppo-learning-rate",
        str(args.ppo_learning_rate),
        "--ppo-ent-coef",
        str(args.ppo_ent_coef),
        "--steane-action-penalty-coef",
        str(args.action_penalty_coef),
        "--steane-miscal-penalty-coef",
        str(args.miscal_penalty_coef),
        "--steane-n-rounds",
        str(args.train_n_rounds),
        "--steane-shots-per-step",
        str(args.train_shots_per_step),
        "--trace-finetune-timesteps",
        str(args.trace_finetune_timesteps),
        "--trace-finetune-rollout-steps",
        str(args.trace_finetune_rollout_steps),
        "--trace-finetune-shots-per-step",
        str(args.trace_finetune_shots_per_step),
        "--trace-finetune-n-rounds",
        str(args.trace_finetune_n_rounds),
        "--post-eval-episodes",
        str(args.post_eval_episodes),
        "--eval-steane-shots-per-step",
        str(args.eval_shots_per_step),
        "--cycle-sweep-rounds",
        str(args.cycle_sweep_rounds),
        "--primary-policy-label",
        "full_channel_RL",
        "--transfer-policy-label",
        "full_channel_transfer_expr1",
        "--transfer-source-checkpoint",
        str(args.transfer_source_checkpoint),
        "--save-primary-policy-checkpoint",
        str(ckpt_path),
        "--save-json",
        str(json_path),
    ]


def shell_line(cmd: list[str]) -> str:
    return "PYTHONPATH=code " + " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = [(seed, build_command(args, seed)) for seed in seeds]
    if args.dry_run:
        for seed, cmd in commands:
            print(f"# seed={seed}")
            print(shell_line(cmd))
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = "code"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    env.setdefault("KMP_AFFINITY", "disabled")
    env.setdefault("KMP_SETTINGS", "0")
    env.setdefault("KMP_WARNINGS", "0")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KMP_USE_SHM", "0")
    pending = list(commands)
    running: list[tuple[int, subprocess.Popen[bytes], Path, object]] = []
    failures: list[int] = []

    while pending or running:
        while pending and len(running) < max(1, int(args.max_workers)):
            seed, cmd = pending.pop(0)
            log_path = out_dir / f"{args.phase_label}_seed{seed}.log"
            log_f = log_path.open("wb")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            running.append((seed, proc, log_path, log_f))
            print(f"launched seed={seed} -> {log_path}")

        next_running: list[tuple[int, subprocess.Popen[bytes], Path, object]] = []
        for seed, proc, log_path, log_f in running:
            rc = proc.poll()
            if rc is None:
                next_running.append((seed, proc, log_path, log_f))
                continue
            log_f.close()
            if rc == 0:
                print(f"completed seed={seed}")
            else:
                print(f"failed seed={seed} rc={rc} log={log_path}")
                failures.append(seed)
        running = next_running
        if pending or running:
            time.sleep(1.0)

    if failures:
        raise SystemExit(f"Expr4 batch finished with failed seeds: {failures}")
    print(f"Expr4 batch finished successfully for seeds={seeds}")


if __name__ == "__main__":
    main()
