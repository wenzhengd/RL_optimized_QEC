# Issue: Correlated Channel Simulation Performance

## Status
- Open
- Priority: Medium-High
- Scope: `correlated_pauli_noise_channel` runtime in Steane RL loops
- Progress:
  - Phase A complete: correlated kernel vectorization + safe event caching
  - Phase B complete: one-pass composed channel executor added for
    `google + correlated` composition
  - Phase C complete (2026-03-09): end-to-end Steane simulator hotpath
    optimization for RL loop runtime

## Problem Summary
`correlated_pauli_noise_channel` is significantly slower than the Google-like channels in practical training/eval runs.

Main reason:
- The correlated model is stateful across idle windows and across `apply(...)` calls within one shot.
- Therefore `apply`-result caching is intentionally disabled (`disable_apply_cache=True`), so repeated sub-circuit noise insertion cannot be memoized.

Related code:
- `code/quantum_simulation/noise_engine.py` (`HiddenMarkovCorrelatedPauliNoiseModel`)
- `code/quantum_simulation/steane_code_simulator.py` (`_ApplyCacheNoiseModel`, cache gating)
- `code/rl_train/steane_adapter.py` (forces `effective_shot_workers=1` for correlated channel)

## Evidence Collected
Small matched benchmark (`eval_steane_ppo`, CPU, tiny config):
- `google_gate_specific`: about `5.83s`
- `correlated_pauli_noise_channel`: about `18.80s`
- Slowdown: about `3.2x`

Simulator-level micro benchmark (`n_steps=12`, `shots=120`):
- `google + apply cache`: `0.099s`
- `google without apply cache`: `8.879s`
- `correlated`: `6.565s`
- Interpretation: current speed gap is dominated by losing full `apply` caching.

`build_events` overhead share inside correlated `apply`:
- Typical repeated stabilizer sub-circuit: around `~6.5%` of `apply` time.
- This implies event-level caching alone likely gives only modest gains.

After Phase A optimization (same local environment):
- simulator micro benchmark (`n_steps=12`, `shots=120`):
  - `correlated`: `6.565s -> 3.749s` (~`1.75x` faster)
- tiny `eval_steane_ppo` benchmark:
  - `correlated`: `18.80s -> 13.72s` (~`1.37x` faster)

After Phase B composition path:
- new channels:
  - `composed_google_global_correlated`
  - `composed_google_gate_specific_correlated`
- apply-level runtime proxy on repeated stabilizer sub-circuit:
  - one-pass composed apply vs separate gate+idle passes:
    `~1.48x` faster (runtime baseline comparison)

## Update (2026-03-09): RL End-to-End Bottleneck Re-Profile + Fixes

### Re-profile conclusion
- The prior `noise_engine` correlated-kernel speedups were real, but RL end-to-end
  runtime was still dominated by simulator-side assembly/execution overhead in
  `steane_code_simulator.py`, not PPO update math.
- Long profile baseline (`eval_steane_ppo`, same fixed config):
  - total: `55.40s`
  - `run_experiment`: `51.26s`
  - `_compile_stream_circuit_plan`: `23.05s`
  - `_make_single_instruction`: `20.58s`
  - `_ppo_update`: `0.12s`

### What was changed
1. `stim` instruction construction hotpath:
   - `code/quantum_simulation/steane_code_simulator.py`
   - `_make_single_instruction(...)` now builds
     `stim.CircuitInstruction(...)` directly, removing per-call temporary
     `stim.Circuit + append` overhead.
2. Cross-run circuit template caching:
   - `code/quantum_simulation/steane_code_simulator.py`
   - added `_cached_shot_templates(...)` with `@lru_cache(maxsize=32)`
   - reused immutable templates (`encoding`, `prep`, stabilizers, rotate, logical
     measurement) across repeated `run_experiment(...)` calls.
3. Minor RL-side overhead trims (small impact):
   - `code/rl_train/ppo.py`
   - `code/rl_train/train.py`
   - `code/rl_train/benchmarks/eval_steane_ppo.py`

### End-to-end timing impact (same command/config)
- Baseline: `47.52s`
- After RL-side-only tweaks: `46.83s`, repeat `47.28s` (small)
- After direct `CircuitInstruction` construction: `31.96s`, repeat `31.52s`
- After template cache: `16.11s`, repeat `16.91s`
- Net vs baseline: about `2.8x` faster end-to-end on this benchmark family.

### Profile delta after latest fixes
- total: `55.40s -> 23.72s`
- `run_experiment`: `51.26s -> 19.59s`
- `_compile_stream_circuit_plan`: `23.05s -> 7.58s`
- `_make_single_instruction`: `20.58s -> 5.35s`
- `measure_single_stabilizer`: `10.57s -> 0.15s`
- `encoding_circuit`: `4.52s -> 0.06s`

### Incremental Round-2 hotspot pass (same day)
Additional targeted changes:
- `code/quantum_simulation/noise_engine.py`
  - removed list conversion in `_sample_idle_events_and_advance` (`flatnonzero -> ndarray`)
- `code/quantum_simulation/steane_code_simulator.py`
  - packed `DEPOLARIZE2` pair targets into one instruction per event
- `code/quantum_simulation/noise_engine.py`
  - matched `DEPOLARIZE2` packing in non-stream/composed builders

Profile comparison (`after3 -> after5`, same benchmark config):
- total: `23.72s -> 22.83s` (~`1.04x`)
- `run_experiment`: `19.59s -> 18.76s`
- `_sample_idle_events_and_advance`: `7.60s -> 6.79s`
- `apply_idle_window_to_simulator`: `7.84s -> 7.39s`
- `_compile_stream_circuit_plan`: `7.58s -> 7.54s` (small additional gain)
- `_make_single_instruction`: `5.35s -> 5.34s` (flat)

### Regression/smoke check
- Correlated-channel tiny smoke (`correlated_pauli_noise_channel`) completed
  successfully after these changes.
- Added hotpath regression tests:
  - file: `code/quantum_simulation/hotpath_regression_test.py`
  - run: `PYTHONPATH=code python -m quantum_simulation.hotpath_regression_test`
  - current status: pass (`4 tests`)

## Update (2026-03-09): Staged Composite Run Re-check (New Backend Only)

Goal:
- Re-run one existing staged composite workload with the current backend and
  compare against the already-saved old-backend artifact.

Compared artifacts:
- old (existing): `code/data_generated/steane_stage89_backend_pilot_old/stage8_scale_x3_backend_pilot/seed_140.json`
- new (this run): `code/data_generated/composite_backend_compare_20260309/stage8_seed140_new_backend.json`
- comparison dump: `code/data_generated/composite_backend_compare_20260309/stage8_seed140_old_new_compare.json`

Timing:
- new backend wall-clock: `890 s` (`14m 50s`)
  - source: `code/data_generated/composite_backend_compare_20260309/stage8_seed140_new_backend_wall_sec.txt`
- old backend wall-clock is not stored inside the old JSON artifact.
  - using the previously stated old baseline (`~30 min`), this implies about
    `2.0x` faster wall-clock on this staged composite setup.

Metric comparison (seed=140, same staged structure, composed correlated channel):
- learned LER~: `0.07031` (old) -> `0.07422` (new)  [delta `+0.00391`]
- learned success: `0.92969` (old) -> `0.92578` (new) [delta `-0.00391`]
- improve(LER~) vs fixed-zero: `+25.00%` (old) -> `+47.22%` (new)

Interpretation:
- Absolute learned-policy quality is nearly unchanged (within ~0.4 percentage
  points in success).
- New backend run completed in `14m50s`; under the old `~30 min` reference,
  runtime is materially better while preserving comparable learned-policy
  behavior.

## Risk
- Larger staged runs can become much longer when switched to correlated channel.
- This can reduce experiment throughput and increase iteration cost.

## Why This Is Correct (and not a bug)
- The correlated model must preserve hidden-state evolution across idle windows and reset only at shot boundaries.
- Reusing a previously generated noisy circuit would break Markov-state semantics and produce incorrect physics/statistics.

## Candidate Optimizations (Future Work)
1. (Done) Reduced Stim construction overhead in stream plan path by replacing
   temporary `Circuit+append` instruction construction with direct
   `stim.CircuitInstruction(...)`.
2. Add optional profiling hooks to report `apply` time breakdown by model and sub-circuit.
3. Revisit parallel strategy for stateful channels:
   use per-worker/per-shot independent noise instances to re-enable shot-level parallelism safely.
4. Add combined-channel benchmark suite (`google + correlated`) to prevent performance regressions.
5. Extend composed executor to support more generic channel stacking patterns beyond current pair.

## Suggested Acceptance Criteria For A Future Fix
1. Preserve correlated-channel semantics (same shot-level statefulness behavior).
2. Keep deterministic reproducibility behavior under fixed seeds.
3. Achieve at least `1.5x` speedup for correlated channel on representative staged/eval workloads.
4. Add a regression benchmark script and record baseline vs improved timings.

## Update (2026-03-09): Stage10 Standard Run (5 seeds, New Backend)

Goal:
- Run a normal stage10 configuration with seed-level parallelism on CPU and
  record wall-clock/runtime behavior.

Run config:
- stage spec: `code/data_generated/steane_stage10_rerun_new_backend_20260309/stage_spec_stage10_5seeds.json`
- seeds: `180-184` (5 seeds)
- parallelism: `--seed-workers 5` (matches 5-core CPU setup)
- output: `code/data_generated/steane_stage10_rerun_new_backend_20260309/`

Timing:
- new backend wall-clock: `1739 s` (`28m 59s`)
  - source: `code/data_generated/steane_stage10_rerun_new_backend_20260309/wall_seconds.txt`
- old backend exact wall-clock for stage10 is not logged in old JSON artifacts.
  - from prior old stage scale-curve file timestamps, stage10 on 5-worker setup
    was estimated at roughly `1.8-2.3 h` total.
  - implied speedup range vs this new run: about `3.7x-4.8x` (estimate).

Metric summary (new backend, 5 seeds):
- improve(LER~) vs fixed-zero: `+44.71% +- 11.76%`
- learned success: `93.96% +- 0.47%`
- summary source: `code/data_generated/steane_stage10_rerun_new_backend_20260309/summary.json`
