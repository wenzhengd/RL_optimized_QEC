# Codex Prompt: Implement `subtask1_stim_qec`

## Scope
Implement only these files in this step:
- `subtask1_stim_qec/simulate_stim.py`
- `subtask1_stim_qec/summarize_measurements.py`

Do not modify other subtasks.

---

## 0) Project intent (research context)

This subtask should simulate **QEC syndrome/detector history** under **time-dependent Pauli error channel**.

Given:
- a code choice
- number of rounds
- a per-round Pauli error schedule

Output:
- history data over time (round-indexed detector/syndrome bits)
- metadata for reproducibility and downstream RL usage

---

## 1) Phase plan (to control complexity)

### Phase A (must do first)
- Code: `small_surface`
- Output granularity: detector-event history
- Noise schedule: per-round `p_x, p_y, p_z`

### Phase B (after A is stable)
- Add `steane7`

### Phase C (optional later)
- Add `shor9`
- Add `gauge_color_15` only if circuit definition is validated and runtime is acceptable

Important:
- Do not start from all codes at once.
- First deliver a stable and validated `small_surface` pipeline.

---

## 2) CLI contract for `simulate_stim.py`

Required arguments:
- `--code` choices: `small_surface`, `steane7`, `shor9`, `gauge_color_15`
- `--rounds` int, `>=1`
- `--shots` int, `>=1`
- `--seed` int
- `--schedule_file` path to JSON/YAML defining per-round Pauli probabilities
- `--out` optional output path

Optional arguments:
- `--distance` int (for surface code), default `3`
- `--save_measurements` flag (if true, also save raw measurement bits)
- `--strict` flag (if true, invalid schedule => hard fail)

Default output path:
- `subtask1_stim_qec/stim_{code}_seed{seed}.npz`

---

## 3) Time-dependent Pauli schedule format (fixed)

Use JSON file. Example:

```json
{
  "version": 1,
  "schedule_type": "per_round",
  "rounds": 5,
  "default": {"p_x": 0.001, "p_y": 0.0, "p_z": 0.001},
  "overrides": {
    "2": {"p_x": 0.005, "p_y": 0.0, "p_z": 0.002}
  }
}
```

Interpretation:
- `default` applies to all rounds unless overridden
- round key is 0-based integer index

Validation rules:
- `p_x, p_y, p_z >= 0`
- `p_x + p_y + p_z <= 1`
- if `schedule.rounds` exists, it must match `--rounds` (or warn if non-strict mode)

---

## 4) What “history granularity” means (fixed decision)

This subtask uses **detector-event history** as default output.

Primary output tensor:
- `detector_history`: shape `(shots, rounds, n_detectors_per_round)`
- binary values `0/1`

Optional raw output (when `--save_measurements`):
- `measurement_history`: shape `(shots, n_measurements_total)`

Why this choice:
- compact
- decoder-friendly
- good for RL features later

---

## 5) Output file format (`.npz`)

Required fields:
- `detector_history` (uint8/int8)
- `time_round` shape `(rounds,)`, values `0..rounds-1`
- `meta_json` (JSON string)

Optional fields:
- `measurement_history` (if `--save_measurements`)

`meta_json` must include:
- `code`, `rounds`, `shots`, `seed`
- `schedule_file`
- `schedule_digest` (hash of schedule content)
- `n_detectors_total`
- `n_detectors_per_round`
- `stim_version` if available

---

## 6) Stim integration requirements

1. Circuit generation:
- Build code-specific stabilizer measurement circuit
- Insert round structure using `TICK` / `REPEAT` where appropriate

2. Time-dependent Pauli injection:
- Apply round-dependent Pauli channel according to schedule
- Keep model Pauli-only in this subtask

3. Sampling:
- Use Stim detector sampler for detector data
- Ensure output can be reshaped into round-indexed history

4. Time-coordinate consistency:
- If using detector coordinates (`SHIFT_COORDS`), ensure round mapping is explicit and reproducible

---

## 7) `summarize_measurements.py` requirements

Input:
- `--input` path to `.npz` from `simulate_stim.py`
- `--out_json` optional
- `--save_plot` optional

Compute and report:
- shape summary
- per-round detector trigger rate (mean over shots)
- global trigger rate
- optional simple temporal correlation over rounds

Outputs:
- terminal summary
- JSON summary (default path: `subtask1_stim_qec/summary_{code}_seed{seed}.json`)
- optional plot: `subtask1_stim_qec/trigger_rate_{code}_seed{seed}.png`

---

## 8) Validation checklist

Must pass:
1. shape consistency:
- `detector_history.shape[0] == shots`
- `detector_history.shape[1] == rounds`

2. binary consistency:
- values are only `0` or `1`

3. reproducibility:
- same seed + same schedule + same args => identical output

4. schedule sanity:
- invalid probabilities trigger clear error

5. runtime sanity (small case):
- `rounds<=5`, `shots<=256` should run quickly on laptop

---

## 9) Canonical smoke commands

1) Small surface code generation:
```bash
python subtask1_stim_qec/simulate_stim.py \
  --code small_surface \
  --distance 3 \
  --rounds 5 \
  --shots 128 \
  --seed 11 \
  --schedule_file subtask1_stim_qec/examples/schedule_small.json
```

2) Summary:
```bash
python subtask1_stim_qec/summarize_measurements.py \
  --input subtask1_stim_qec/stim_small_surface_seed11.npz \
  --save_plot
```

---

## 10) Dependency policy

Required:
- `stim`
- `numpy`
- `matplotlib` (for optional summary plot)

Optional:
- `pyyaml` only if YAML schedule support is implemented

---

## 11) Beginner-friendly coding style

- Add comments for key steps:
  - schedule validation
  - round-dependent error injection
  - detector history reshaping
- Keep function boundaries simple and explicit
- Avoid heavy framework abstractions



