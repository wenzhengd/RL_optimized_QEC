"""Subtask-0: inspect generated noise and validate intended characteristics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, skew as scipy_skew


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for noise inspection."""
    parser = argparse.ArgumentParser(description="Inspect generated noise file.")
    # Input and output controls.
    parser.add_argument("--input", type=str, required=True, help="Input .npz noise file.")
    parser.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional output summary JSON path.",
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Save histogram/ACF/PSD plots.",
    )
    parser.add_argument(
        "--lag_max",
        type=int,
        default=100,
        help="Maximum lag used for ACF plots/metrics.",
    )
    return parser.parse_args()


def _parse_meta(meta_json_obj: Any) -> dict[str, Any]:
    """Parse metadata JSON stored in npz."""
    if isinstance(meta_json_obj, np.ndarray):
        raw = meta_json_obj.item()
    else:
        raw = meta_json_obj
    return json.loads(raw)


def _acf_ensemble(noise: np.ndarray, lag_max: int) -> np.ndarray:
    """Compute ensemble-averaged autocorrelation up to lag_max."""
    n_traj, n_time = noise.shape
    lag_max = min(lag_max, n_time - 1)
    # Center each trajectory first, then average correlations.
    centered = noise - noise.mean(axis=1, keepdims=True)
    var = np.mean(centered**2)
    if var <= 0:
        return np.ones(lag_max + 1)

    acf = np.empty(lag_max + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, lag_max + 1):
        prod = centered[:, :-lag] * centered[:, lag:]
        acf[lag] = np.mean(prod) / var
    return acf


def _empirical_psd(noise: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute ensemble-averaged one-sided PSD using FFT."""
    centered = noise - noise.mean(axis=1, keepdims=True)
    # One-sided real FFT is enough for real-valued time signals.
    spec = np.fft.rfft(centered, axis=1)
    psd = np.mean((np.abs(spec) ** 2) / noise.shape[1], axis=0)
    freqs = np.fft.rfftfreq(noise.shape[1], d=dt)
    return freqs, psd


def _target_stats(meta: dict[str, Any]) -> dict[str, float | None]:
    """Compute intended mean/variance/skewness from metadata when defined."""
    model = meta["noise_model"]
    amp = float(meta.get("amplitude", 1.0))
    if model == "iid":
        dist = meta.get("dist", "normal")
        if dist == "normal":
            sigma = float(meta.get("sigma", 1.0))
            return {"mean": 0.0, "var": (amp * sigma) ** 2, "skew": 0.0}
        low = float(meta.get("low", -1.0))
        high = float(meta.get("high", 1.0))
        mean = amp * (low + high) / 2.0
        var = (amp**2) * ((high - low) ** 2) / 12.0
        return {"mean": mean, "var": var, "skew": 0.0}

    if model == "gaussian":
        return {"mean": float(meta.get("dc_offset", 0.0)), "var": None, "skew": 0.0}

    if model == "telegraph":
        p = float(meta.get("p_init_plus", 0.5))
        mean = 2.0 * p - 1.0
        var = max(0.0, 1.0 - mean**2)
        return {"mean": mean, "var": var, "skew": None}

    return {"mean": None, "var": None, "skew": None}


def _normalized_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Return normalized L2 mismatch."""
    denom = np.linalg.norm(b) + 1e-12
    return float(np.linalg.norm(a - b) / denom)


def _status(ok: bool, small_sample: bool) -> str:
    """Convert boolean check into PASS/WARN/FAIL."""
    if ok:
        return "PASS"
    return "WARN" if small_sample else "FAIL"


def _default_out_json(model: str, seed: int) -> Path:
    return Path("subtask0_noise") / f"inspect_{model}_seed{seed}.json"


def _default_plot_path(kind: str, model: str, seed: int) -> Path:
    return Path("subtask0_noise") / f"{kind}_{model}_seed{seed}.png"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    data = np.load(input_path, allow_pickle=True)
    noise = np.asarray(data["noise"], dtype=np.float64)
    time = np.asarray(data["time"], dtype=np.float64)
    meta = _parse_meta(data["meta_json"])

    model = str(meta["noise_model"])
    seed = int(meta.get("seed", -1))
    dt = float(meta.get("dt", time[1] - time[0]))
    amp = float(meta.get("amplitude", 1.0))

    # Flatten so global distribution statistics are easy to compute.
    flat = noise.reshape(-1)
    n_total = int(flat.size)
    small_sample = n_total < 10_000

    mean_emp = float(np.mean(flat))
    var_emp = float(np.var(flat))
    skew_emp = float(scipy_skew(flat, bias=False))

    targets = _target_stats(meta)
    mean_t = targets["mean"]
    var_t = targets["var"]
    skew_t = targets["skew"]

    # Tolerance rules come from the agreed _human_note.md spec.
    mean_ok = True if mean_t is None else abs(mean_emp - mean_t) <= (0.05 * amp + 1e-12)
    var_ok = True
    if var_t is not None and abs(var_t) > 1e-12:
        var_ok = abs(var_emp - var_t) / abs(var_t) <= 0.10
    skew_ok = True if skew_t is None else abs(skew_emp - skew_t) <= 0.20

    lag_max = min(args.lag_max, noise.shape[1] - 1)
    acf_emp = _acf_ensemble(noise, lag_max)
    acf_mismatch = None
    acf_status = "N/A"
    if model == "iid":
        # IID target: near-zero autocorrelation at nonzero lags.
        tail = acf_emp[1 : min(21, len(acf_emp))]
        iid_ok = bool(np.max(np.abs(tail)) <= 0.10)
        acf_status = _status(iid_ok, small_sample)
        acf_mismatch = float(np.max(np.abs(tail)))
    elif model == "telegraph":
        # Telegraph target ACF: exponential decay trend.
        flip_rate = float(meta.get("flip_rate", 0.0))
        lags = np.arange(len(acf_emp), dtype=np.float64)
        acf_target = np.exp(-2.0 * flip_rate * dt * lags)
        acf_mismatch = _normalized_l2(acf_emp[1:], acf_target[1:])
        acf_status = _status(acf_mismatch <= 0.20, small_sample)

    freqs, psd_emp = _empirical_psd(noise, dt)
    psd_mismatch = None
    f_peak_emp = float(freqs[int(np.argmax(psd_emp))])
    f_peak_target = None
    psd_status = "N/A"

    target_psd = None
    if model == "gaussian":
        # Rebuild target spectrum from metadata and compare to empirical PSD.
        f0 = float(meta.get("f0", 0.0))
        sigma_f = float(meta.get("sigma_f", 1.0))
        amp_g = float(meta.get("amplitude", 1.0))
        target_psd = (amp_g**2) * np.exp(-0.5 * ((freqs - f0) / sigma_f) ** 2)

        # Compare normalized shapes so amplitude scaling is not the main factor.
        psd_emp_n = psd_emp / (np.max(psd_emp) + 1e-12)
        psd_tar_n = target_psd / (np.max(target_psd) + 1e-12)
        psd_mismatch = _normalized_l2(psd_emp_n, psd_tar_n)

        f_peak_target = f0
        if f0 > 0:
            peak_rel_err = abs(f_peak_emp - f0) / max(f0, 1e-12)
        else:
            peak_rel_err = abs(f_peak_emp - f0)

        psd_ok = (psd_mismatch <= 0.25) and (peak_rel_err <= 0.15)
        psd_status = _status(psd_ok, small_sample)

    summary: dict[str, Any] = {
        "input_file": str(input_path),
        "noise_model": model,
        "seed": seed,
        "shape": list(noise.shape),
        "mean_empirical": mean_emp,
        "mean_target": mean_t,
        "mean_status": _status(mean_ok, small_sample),
        "var_empirical": var_emp,
        "var_target": var_t,
        "var_status": _status(var_ok, small_sample),
        "skew_empirical": skew_emp,
        "skew_target": skew_t,
        "skew_status": _status(skew_ok, small_sample),
        "acf_mismatch": acf_mismatch,
        "acf_status": acf_status,
        "psd_mismatch": psd_mismatch,
        "psd_status": psd_status,
        "peak_freq_empirical": f_peak_emp,
        "peak_freq_target": f_peak_target,
    }

    print("[Noise Inspection Summary]")
    print(json.dumps(summary, indent=2))

    out_json = Path(args.out_json) if args.out_json else _default_out_json(model, seed)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[OK] Saved summary JSON: {out_json}")

    if args.save_plot:
        # Histogram plot
        hist_path = _default_plot_path("hist_compare", model, seed)
        plt.figure(figsize=(7, 4))
        plt.hist(flat, bins=80, density=True, alpha=0.6, label="Empirical")
        xs = np.linspace(np.min(flat), np.max(flat), 400)
        if model == "iid" and meta.get("dist", "normal") == "normal":
            sigma = float(meta.get("sigma", 1.0)) * amp
            plt.plot(xs, norm.pdf(xs, loc=0.0, scale=sigma), label="Target PDF", lw=2)
        elif model == "iid" and meta.get("dist") == "uniform":
            low = amp * float(meta.get("low", -1.0))
            high = amp * float(meta.get("high", 1.0))
            ys = np.where((xs >= low) & (xs <= high), 1.0 / max(high - low, 1e-12), 0.0)
            plt.plot(xs, ys, label="Target PDF", lw=2)
        plt.title(f"Histogram Comparison ({model})")
        plt.xlabel("Noise value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(hist_path, dpi=140)
        plt.close()

        # ACF plot
        acf_path = _default_plot_path("acf_compare", model, seed)
        lags = np.arange(len(acf_emp))
        plt.figure(figsize=(7, 4))
        plt.plot(lags, acf_emp, label="Empirical ACF")
        if model == "iid":
            plt.plot(lags, np.where(lags == 0, 1.0, 0.0), "--", label="Target ACF")
        elif model == "telegraph":
            flip_rate = float(meta.get("flip_rate", 0.0))
            acf_target = np.exp(-2.0 * flip_rate * dt * lags)
            plt.plot(lags, acf_target, "--", label="Target ACF")
        plt.title(f"ACF Comparison ({model})")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(acf_path, dpi=140)
        plt.close()

        # PSD plot
        psd_path = _default_plot_path("psd_compare", model, seed)
        plt.figure(figsize=(7, 4))
        plt.plot(freqs, psd_emp, label="Empirical PSD")
        if target_psd is not None:
            plt.plot(freqs, target_psd, "--", label="Target PSD")
        plt.title(f"PSD Comparison ({model})")
        plt.xlabel("Frequency")
        plt.ylabel("PSD (a.u.)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(psd_path, dpi=140)
        plt.close()

        print(f"[OK] Saved plots: {hist_path}, {acf_path}, {psd_path}")


if __name__ == "__main__":
    main()
