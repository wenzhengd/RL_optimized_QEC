"""Subtask-0: generate noise trajectories for downstream simulation.

The script writes one `.npz` file containing:
- `noise` with shape (n_traj, n_time)
- `time` with shape (n_time,)
- `meta_json` with all generation arguments
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LUCKY_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for noise generation."""
    parser = argparse.ArgumentParser(description="Generate noise trajectories.")
    # Common controls shared by all models.
    parser.add_argument(
        "--noise_model",
        required=True,
        choices=["iid", "gaussian", "telegraph"],
        help="Noise type to generate.",
    )
    parser.add_argument("--T", type=float, required=True, help="Sampling window [0, T].")
    parser.add_argument("--n_time", type=int, required=True, help="Number of time samples.")
    parser.add_argument(
        "--n_traj",
        type=int,
        default=1000,
        help="Number of noise trajectories in the ensemble.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=LUCKY_SEED,
        help=f"Random seed (forced to lucky seed {LUCKY_SEED}).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Global amplitude scale (except telegraph which is fixed to +/-1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output .npz path. If omitted, a default path is used.",
    )

    # IID-specific parameters.
    parser.add_argument(
        "--dist",
        type=str,
        default="normal",
        choices=["normal", "uniform"],
        help="Distribution for iid noise.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Std dev for iid normal noise.",
    )
    parser.add_argument("--low", type=float, default=-1.0, help="Lower bound for iid uniform.")
    parser.add_argument("--high", type=float, default=1.0, help="Upper bound for iid uniform.")

    # Gaussian-spectrum parameters (target PSD peak and bandwidth).
    parser.add_argument("--f0", type=float, default=1.0, help="Target spectrum peak frequency.")
    parser.add_argument("--sigma_f", type=float, default=1.0, help="Target spectrum bandwidth.")
    parser.add_argument("--dc_offset", type=float, default=0.0, help="Constant offset in time.")

    # Telegraph process parameters (binary +/-1 with random flips).
    parser.add_argument("--flip_rate", type=float, default=1.0, help="Poisson switching rate.")
    parser.add_argument(
        "--p_init_plus",
        type=float,
        default=0.5,
        help="Initial probability for +1 in telegraph noise.",
    )
    return parser.parse_args()


def _default_output_path(noise_model: str, seed: int) -> Path:
    """Build the default output path in subtask0_noise/subtask0_data/."""
    _ = seed
    return Path("subtask0_noise") / "subtask0_data" / f"noise_{noise_model}.npz"


def _validate_args(args: argparse.Namespace) -> float:
    """Validate arguments and return dt."""
    if args.T <= 0:
        raise ValueError("`--T` must be > 0.")
    if args.n_time < 2:
        raise ValueError("`--n_time` must be >= 2.")
    if args.n_traj < 1:
        raise ValueError("`--n_traj` must be >= 1.")

    # Uniform discrete time-step implied by [0, T] and n_time samples.
    dt = args.T / (args.n_time - 1)

    if args.noise_model == "iid":
        if args.dist == "normal" and args.sigma <= 0:
            raise ValueError("For iid normal, `--sigma` must be > 0.")
        if args.dist == "uniform" and not (args.low < args.high):
            raise ValueError("For iid uniform, require `--low < --high`.")

    if args.noise_model == "gaussian":
        if args.sigma_f <= 0:
            raise ValueError("For gaussian spectrum mode, `--sigma_f` must be > 0.")
        if args.f0 < 0:
            raise ValueError("For gaussian spectrum mode, `--f0` must be >= 0.")
        # Nyquist frequency is the max resolvable frequency on this grid.
        f_nyquist = 1.0 / (2.0 * dt)
        if args.f0 > f_nyquist:
            raise ValueError(
                f"`--f0`={args.f0} exceeds Nyquist={f_nyquist:.6g}. "
                "Increase n_time or reduce f0."
            )
        if args.f0 + 3.0 * args.sigma_f > f_nyquist:
            print(
                "[WARN] f0 + 3*sigma_f exceeds Nyquist; target spectrum may be truncated."
            )

    if args.noise_model == "telegraph":
        if args.flip_rate < 0:
            raise ValueError("For telegraph mode, `--flip_rate` must be >= 0.")
        if not (0.0 <= args.p_init_plus <= 1.0):
            raise ValueError("`--p_init_plus` must be in [0, 1].")
        if args.amplitude != 1.0:
            raise ValueError(
                "Telegraph noise is defined as binary +/-1 in this project; "
                "use `--amplitude 1.0`."
            )

    return dt


def _gen_iid(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """Generate iid noise with chosen distribution."""
    shape = (args.n_traj, args.n_time)
    if args.dist == "normal":
        base = rng.normal(loc=0.0, scale=args.sigma, size=shape)
    else:
        base = rng.uniform(low=args.low, high=args.high, size=shape)
    # Apply global scale at the end for consistent handling across modes.
    return args.amplitude * base


def _gen_gaussian_spectrum(
    args: argparse.Namespace,
    rng: np.random.Generator,
    dt: float,
) -> np.ndarray:
    """Generate Gaussian process via frequency-domain filtering."""
    # Start from white noise, then shape its spectrum by a Gaussian envelope.
    white = rng.normal(0.0, 1.0, size=(args.n_traj, args.n_time))
    freqs = np.fft.rfftfreq(args.n_time, d=dt)

    # One-sided target spectrum envelope (shape-only target for inspection).
    target_psd = (args.amplitude**2) * np.exp(
        -0.5 * ((freqs - args.f0) / args.sigma_f) ** 2
    )
    filt = np.sqrt(target_psd + 1e-18)

    white_f = np.fft.rfft(white, axis=1)
    shaped_f = white_f * filt[np.newaxis, :]
    # Return to time domain after spectral shaping.
    noise = np.fft.irfft(shaped_f, n=args.n_time, axis=1)

    if args.dc_offset != 0.0:
        noise = noise + args.dc_offset
    return noise


def _gen_telegraph(
    args: argparse.Namespace,
    rng: np.random.Generator,
    dt: float,
) -> np.ndarray:
    """Generate telegraph noise with Poisson-distributed switching events."""
    n_traj, n_time = args.n_traj, args.n_time
    # Convert continuous flip rate to discrete per-step flip probability.
    p_flip = 1.0 - np.exp(-args.flip_rate * dt)

    noise = np.empty((n_traj, n_time), dtype=np.float64)
    init_plus = rng.random(n_traj) < args.p_init_plus
    noise[:, 0] = np.where(init_plus, 1.0, -1.0)

    flips = rng.random((n_traj, n_time - 1)) < p_flip
    for t in range(1, n_time):
        prev = noise[:, t - 1]
        # Flip sign on event, otherwise keep previous state.
        noise[:, t] = np.where(flips[:, t - 1], -prev, prev)
    return noise


def main() -> None:
    args = parse_args()
    # Project decision: use one fixed lucky seed for all task0 generations.
    if args.seed != LUCKY_SEED:
        print(f"[WARN] Overriding --seed={args.seed} to lucky seed {LUCKY_SEED}.")
    args.seed = LUCKY_SEED

    dt = _validate_args(args)
    rng = np.random.default_rng(args.seed)

    # Store explicit time grid so downstream scripts can reuse it directly.
    time = np.linspace(0.0, args.T, args.n_time)

    if args.noise_model == "iid":
        noise = _gen_iid(args, rng)
    elif args.noise_model == "gaussian":
        noise = _gen_gaussian_spectrum(args, rng, dt)
    else:
        noise = _gen_telegraph(args, rng, dt)

    out_path = Path(args.out) if args.out else _default_output_path(args.noise_model, args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "noise_model": args.noise_model,
        "seed": args.seed,
        "T": args.T,
        "n_time": args.n_time,
        "dt": dt,
        "n_traj": args.n_traj,
        "amplitude": args.amplitude,
        "dist": args.dist,
        "sigma": args.sigma,
        "low": args.low,
        "high": args.high,
        "f0": args.f0,
        "sigma_f": args.sigma_f,
        "dc_offset": args.dc_offset,
        "flip_rate": args.flip_rate,
        "p_init_plus": args.p_init_plus,
    }

    np.savez_compressed(
        out_path,
        noise=noise.astype(np.float64),
        time=time.astype(np.float64),
        meta_json=json.dumps(meta),
    )
    print(f"[OK] Saved noise file: {out_path}")
    print(f"[INFO] noise shape: {noise.shape}")


if __name__ == "__main__":
    main()
