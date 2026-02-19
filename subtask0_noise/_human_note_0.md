# Codex Prompt: Implement `subtask0_noise`

## Scope
Implement only these two files:
- `subtask0_noise/generate_noise.py`
- `subtask0_noise/inspect_noise.py`

Do not modify other subtasks in this step.

## Goal
Create a noise trajectory generator `f(t)` and an inspection script for basic statistics.

---

## 1) `generate_noise.py` requirements

### 1.1 Supported noise types
1. `iid`
- Independent samples in time.
- Default distribution: standard normal (unless CLI adds alternatives later).

2. `gaussian`
- Zero-mean Gaussian process in time.
- User specifies a Gaussian-shaped spectrum in frequency domain:
  - spectral peak frequency `f0`
  - spectral bandwidth `sigma_f`

3. `telegraph`
- Two-level process: `f(t) in {+1, -1}`.
- Initial state is zero-mean by design:
  - `P(f(0)=+1)=0.5`, `P(f(0)=-1)=0.5`
- State flips follow a Poisson process with rate `lambda` (user-specified).

### 1.2 Common sampling controls (all noise types)
User must be able to specify:
- time range `[0, T]`
- number of uniform time samples `n_time`
- ensemble size `n_traj` (default `1000`)
- random seed
- global amplitude scale `amplitude` (default `1.0`)

Derived quantity:
- `dt = T / (n_time - 1)` (must be well-defined, so require `n_time >= 2`)

### 1.3 CLI contract (minimum)
Use argument names that are explicit and stable:
- `--noise_model` in `{iid, gaussian, telegraph}`
- `--T` (float)
- `--n_time` (int)
- `--n_traj` (int, default `1000`)
- `--seed` (int)
- `--out` (output `.npz` path; optional default naming is fine)

Additional by model:
- `gaussian`: `--f0`, `--sigma_f`, optional `--amplitude`
- `telegraph`: `--flip_rate`
- `iid`: `--dist` in `{normal, uniform}` and distribution parameters
  - if `normal`: `--sigma`
  - if `uniform`: `--low`, `--high`

Recommended additional model options:
- `gaussian`: optional `--dc_offset` (default `0.0`)
- `telegraph`: optional `--p_init_plus` (default `0.5`)

### 1.3.1 Validity checks (must enforce)
Common:
- `T > 0`
- `n_time >= 2`
- `n_traj >= 1`

IID:
- normal mode: `sigma > 0`
- uniform mode: `low < high`

Gaussian spectrum mode:
- `sigma_f > 0`
- `f0 >= 0`
- `f_nyquist = 1 / (2*dt)`
- require `f0 <= f_nyquist`
- recommended warning/check: `f0 + 3*sigma_f <= f_nyquist` to avoid strong truncation

Telegraph:
- `flip_rate >= 0`
- transition probability per step:
  - `p_flip = 1 - exp(-flip_rate * dt)`

### 1.4 Output file format (`.npz`)
Required arrays/fields:
- `noise`: shape `(n_traj, n_time)`
- `time`: shape `(n_time,)`
- `meta_json`: JSON string with all generation parameters

`meta_json` should include enough information for full reproducibility and spectrum checks:
- `noise_model`, `seed`, `T`, `n_time`, `dt`, `n_traj`, `amplitude`
- model-specific parameters (`f0`, `sigma_f`, `flip_rate`, `dist`, etc.)

---

## 2) `inspect_noise.py` requirements

Given a generated noise file, report statistical properties of the whole dataset:
- mean
- variance
- skewness

Use a standard Python statistics/scientific library for skewness (e.g. `scipy.stats.skew`), with a fallback if unavailable.

### 2.1 CLI contract (minimum)
- `--input` path to `.npz` noise file

### 2.2 Output
- Print a clear text summary:
  - model type (from metadata)
  - shape `(n_traj, n_time)`
  - mean / variance / skewness
- Optional: save summary to JSON if `--out_json` is provided.

### 2.3 FFT/PSD check for Gaussian-spectrum noise (new requirement)
If `noise_model == gaussian` and metadata includes frequency-domain parameters:
1. Compute empirical spectrum from generated trajectories:
- FFT each trajectory along time axis
- compute PSD
- average PSD over ensemble

2. Reconstruct target spectrum from metadata:
- use stored `f0`, `sigma_f`, `amplitude` (and optional `dc_offset`)
- target shape example:
  - `S_target(f) = A * exp(-(f - f0)^2 / (2*sigma_f^2))`

3. Compare empirical vs target:
- report peak frequency (target vs empirical)
- report a mismatch metric (e.g., normalized L2 error)
- optional: save a comparison plot if `--save_plot` is provided

### 2.4 Intended-vs-generated comparison (plot + metrics)
`inspect_noise.py` should compare "intended noise characteristics" vs "generated noise characteristics".

Use the following compact, robust set:
1. Time-domain distribution check (all models)
- Compare target mean/variance/skewness against empirical values.
- Plot histogram of generated samples (flattened over trajectory/time).
- If a theoretical PDF exists (e.g., normal IID), overlay it.

2. Temporal correlation check
- Compute and plot autocorrelation function (ACF) for generated noise.
- For IID, expected ACF is near zero for nonzero lag.
- For telegraph, compare empirical ACF against expected exponential decay trend.

3. Frequency-domain check
- Plot empirical PSD (ensemble-averaged FFT-based PSD).
- If a target spectrum is defined (Gaussian mode), overlay target PSD on same axes.
- Report mismatch score (normalized L2 error or similar).

Recommended plot outputs:
- `hist_compare.png`
- `acf_compare.png`
- `psd_compare.png`

Recommended numeric summary fields in JSON/text:
- `mean_empirical`, `mean_target`
- `var_empirical`, `var_target`
- `skew_empirical`, `skew_target` (if target known)
- `acf_mismatch` (optional)
- `psd_mismatch` (for Gaussian spectrum mode)
- `peak_freq_empirical`, `peak_freq_target` (for Gaussian spectrum mode)

---

## 3) Numerical sanity checks

At minimum, implementation should satisfy:
1. Reproducibility:
- same seed + same args => same output noise.

2. Telegraph constraints:
- values are only `+1` or `-1`.

3. Gaussian process behavior:
- near-zero mean for large enough ensemble.
- skewness near zero.
- for frequency-domain Gaussian mode: empirical PSD approximately matches target PSD.

4. Shape consistency:
- output exactly matches `(n_traj, n_time)`.

---

## 4) Style constraints
- Keep code simple and educational.
- Add comments aimed at a beginner developer.
- Avoid over-engineering and unnecessary abstractions.

---

## 5) Output locations and naming (fixed)

All generated/inspection artifacts for this subtask must stay inside `subtask0_noise/`.

Default paths:
- Generated noise file:
  - `subtask0_noise/noise_{noise_model}_seed{seed}.npz`
- Inspection summary JSON:
  - `subtask0_noise/inspect_{noise_model}_seed{seed}.json`
- Inspection plots:
  - `subtask0_noise/hist_compare_{noise_model}_seed{seed}.png`
  - `subtask0_noise/acf_compare_{noise_model}_seed{seed}.png`
  - `subtask0_noise/psd_compare_{noise_model}_seed{seed}.png`

If user provides explicit output arguments, use user-provided paths.

---

## 6) Default validation tolerances (good practical defaults)

Use these as pass/fail defaults when reporting `PASS/WARN/FAIL` in `inspect_noise.py`.

For large enough samples (`n_traj * n_time >= 1e4`):
- mean tolerance:
  - `abs(mean_empirical - mean_target) <= 0.05 * amplitude + 1e-12`
- variance tolerance:
  - relative error `<= 0.10`
- skewness tolerance:
  - `abs(skew_empirical - skew_target) <= 0.20` (or `abs(skew_empirical) <= 0.20` when target is 0)

IID ACF (nonzero lags up to lag 20):
- `max(abs(acf_empirical[1:21])) <= 0.10`

Telegraph ACF:
- compare to exponential trend; normalized L2 mismatch `<= 0.20`

Gaussian PSD:
- normalized L2 mismatch `<= 0.25`
- peak frequency relative error:
  - `abs(f_peak_emp - f_peak_target) / max(f_peak_target, 1e-12) <= 0.15`

If sample count is small, allow `WARN` instead of hard `FAIL`.

---

## 7) Canonical test cases (fixed seeds)

Use these three commands as baseline checks after implementation.

1. IID normal test:
```bash
python subtask0_noise/generate_noise.py \
  --noise_model iid --dist normal --sigma 1.0 \
  --T 10.0 --n_time 2048 --n_traj 128 --seed 101
python subtask0_noise/inspect_noise.py \
  --input subtask0_noise/noise_iid_seed101.npz
```

2. Gaussian-spectrum test:
```bash
python subtask0_noise/generate_noise.py \
  --noise_model gaussian --f0 5.0 --sigma_f 1.0 --amplitude 1.0 \
  --T 10.0 --n_time 2048 --n_traj 128 --seed 202
python subtask0_noise/inspect_noise.py \
  --input subtask0_noise/noise_gaussian_seed202.npz --save_plot
```

3. Telegraph test:
```bash
python subtask0_noise/generate_noise.py \
  --noise_model telegraph --flip_rate 2.0 --p_init_plus 0.5 \
  --T 10.0 --n_time 2048 --n_traj 128 --seed 303
python subtask0_noise/inspect_noise.py \
  --input subtask0_noise/noise_telegraph_seed303.npz --save_plot
```

---

## 8) Dependency policy (confirmed)

Confirmed runtime dependencies:
- `numpy` (required)
- `scipy` (required; stats and signal utilities)
- `matplotlib` (required for plots)

No optional fallback path is required for these libraries in this subtask.

---

## 9) Validation command checklist (for user)

After implementation, run in this order:

1. Syntax check:
```bash
python -m py_compile subtask0_noise/generate_noise.py subtask0_noise/inspect_noise.py
```

2. Run all canonical test cases (Section 7).

3. Confirm outputs exist in `subtask0_noise/`:
- `.npz` generated files
- `.json` summary files (if enabled)
- `.png` plots (when `--save_plot` is used)

4. Confirm inspector prints clear pass/warn/fail per metric:
- mean
- variance
- skewness
- ACF check
- PSD check (for Gaussian spectrum mode)
