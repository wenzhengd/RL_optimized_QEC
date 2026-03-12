# Handoff Notes (Correlated + Composed Channel Work)

## Done
1. Phase A complete: correlated-channel speed optimization
- File: `code/quantum_simulation/noise_engine.py`
- Main changes:
  - Vectorized correlated idle sampling/state update
  - 2-state symmetric transition fast-path
  - Safe events cache (deterministic structure only)

2. Phase B complete: one-pass composed channel path
- New composed channels:
  - `composed_google_gate_specific_correlated`
  - `composed_google_global_correlated`
- Files:
  - `code/quantum_simulation/noise_engine.py`
    - `ComposedGateAndCorrelatedIdleNoiseModel`
  - `code/quantum_simulation/noise_channels.py`
    - channel registry + factory construction
  - `code/rl_train/steane_adapter.py`
    - stateful detection via `noise.stateful` -> force effective `shot_workers=1`
  - `code/rl_train/train.py`
  - `code/rl_train/benchmarks/eval_steane_ppo.py`
    - CLI channel choices updated
  - `code/rl_train/README.md`
    - usage docs updated

3. Tracking updated
- File: `issue.md`
  - progress + benchmark notes updated

## Key Behavior
- Staged runs done previously were effectively `google_gate_specific` (default `auto + gate_specific` path).
- New composed channel corresponds to “gate-specific (action-dependent) + correlated idle (native/time-correlated)” in one pass.

## Smoke Commands
1. Composed channel tiny smoke:
```bash
env OMP_NUM_THREADS=1 PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python -m rl_train.benchmarks.eval_steane_ppo \
  --total-timesteps 8 --rollout-steps 2 --max-steps 1 \
  --steane-n-rounds 1 --steane-shots-per-step 2 \
  --post-eval-episodes 1 --eval-steane-shots-per-step 2 \
  --trace-finetune-timesteps 0 --device cpu \
  --steane-noise-channel composed_google_gate_specific_correlated \
  --steane-shot-workers 1 \
  --save-json /tmp/eval_composed_smoke.json
```

2. Correlated tiny smoke:
```bash
env OMP_NUM_THREADS=1 PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python -m rl_train.benchmarks.eval_steane_ppo \
  --total-timesteps 16 --rollout-steps 4 --max-steps 1 \
  --steane-n-rounds 2 --steane-shots-per-step 8 \
  --post-eval-episodes 2 --eval-steane-shots-per-step 8 \
  --trace-finetune-timesteps 0 --device cpu \
  --steane-noise-channel correlated_pauli_noise_channel \
  --steane-shot-workers 1 \
  --save-json /tmp/eval_corr_small_after_opt.json
```

## Not Done Yet
- Phase C intentionally skipped:
  - Re-enabling shot-level parallelism for stateful channels via per-worker noise instances.

## Session Theme (2026-03-09)
- Main theme:
  - `noise_circuit_simulation` (especially correlated/composed channels) has been substantially accelerated.
  - End-to-end RL remains slow, so total runtime improvement can look small in tiny RL benchmarks.

- What was confirmed:
  - Kernel-level old/new benchmark (same workload, backend switch only):
    - `correlated_pauli_noise_channel`: `4.759s -> 0.437s` (~`10.89x`)
    - `composed_google_gate_specific_correlated`: `10.670s -> 0.597s` (~`17.87x`)
  - Quick RL benchmark showed only `95s -> 85s`:
    - This is interpreted as RL/training/eval overhead dominating wall-clock at that budget,
      not as a failure of simulator-kernel optimization.

- Important backend switch:
  - Legacy path: `STEANE_DISABLE_STREAMING_NOISE=1`
  - New fast path (default): `STEANE_DISABLE_STREAMING_NOISE=0`

- Next Codex session priority:
  - Focus optimization on RL-side bottlenecks (outside noisy-circuit kernel), e.g.:
    - PPO/train loop overhead
    - eval frequency and policy-eval pipeline overhead
    - process/thread orchestration and serialization overhead
    - non-simulator Python hotspots in benchmark driver

## Session Theme (2026-03-11)

### Done
1. Measurement-error overlay implemented
- Added symmetric pre-measurement bit-flip model as an overlay, not a standalone noise-channel key.
- Main files:
  - `code/quantum_simulation/noise_engine.py`
    - `PreMeasurementBitFlipNoiseModel`
  - `code/quantum_simulation/noise_channels.py`
    - measurement overlay wrapped around existing channels
  - `code/quantum_simulation/steane_code_simulator.py`
    - stream path inserts pre-measurement noise before `M`
  - `code/rl_train/steane_adapter.py`
  - `code/rl_train/train.py`
  - `code/rl_train/benchmarks/eval_steane_ppo.py`

2. Timing summary helpers added
- `code/quantum_simulation/noise_engine.py`
  - `CircuitTimingSummary`
  - `summarize_circuit_timing(...)`
- `code/rl_train/steane_adapter.py`
  - nominal timing exported in step `info`
- `code/rl_train/benchmarks/eval_steane_ppo.py`
  - benchmark JSON now includes `nominal_circuit_timing_per_rl_step`

3. Documentation and QA updated
- `code/quantum_simulation/noise_modeling.md` added
- `code/rl_train/README.md` updated to link/document noise modeling
- `human_codex_QA.md` updated with:
  - RL config surfaces
  - correlated-channel `f,g` semantics
  - circuit timing semantics
  - measurement overlay timing overhead
  - `n_rounds / n_steps / RL step` clarification

4. Experiment-plan README upgraded
- File:
  - `code/data_generated/rl_Steane_tune_experiment/README.md`
- Current state:
  - 4 experiments defined
  - `Experiment 4` added:
    - full composite
    - fixed-policy evaluation sweep
    - `n_rounds = 5,10,...,50`
    - showcase condition fixed to:
      - `f = 1e2`
      - `g = 0.1`
      - `p_meas = 0.01`
  - README also now includes:
    - shots-per-candidate clarification
    - recommended execution order
    - A/B/C phase structure

5. New cycle-sweep driver added
- New file:
  - `code/rl_train/benchmarks/eval_steane_cycle_sweep.py`
- Purpose:
  - train one policy once
  - hold it fixed
  - evaluate over `n_rounds` sweep
  - report:
    - `success_rate`
    - `ler_proxy`
    - `logical_observable_proxy = 2*success_rate - 1`
    - improvement vs `fixed_zero`
- Supports optional second trained policy via:
  - `--secondary-steane-measurement-bitflip-prob`
- Also refactored:
  - `code/rl_train/benchmarks/eval_steane_ppo.py`
    - added reusable `build_arg_parser()`

6. Expr1-A stage spec prepared
- New file:
  - `code/rl_train/benchmarks/examples/stage_specs_expr1_phaseA_gate_quick.json`
- Contents:
  - 6 gate-only quick-scan conditions
  - two ratio slices:
    - `1q/2q = 1/4`
    - `1q/2q = 4/1`
  - three scales each:
    - `0.5`
    - `1.0`
    - `1.5`
  - 5 seeds per condition
  - budget per run:
    - `total_timesteps = 512`
    - `rollout_steps = 32`
    - `steane_n_rounds = 4`
    - `steane_shots_per_step = 4`
    - `post_eval_episodes = 8`
    - `eval_steane_shots_per_step = 24`

### Verified
- Static compile passed:
```bash
python -m py_compile \
  code/rl_train/benchmarks/eval_steane_ppo.py \
  code/rl_train/benchmarks/eval_steane_cycle_sweep.py
```

### Important Runtime Blocker
- Attempting to run `Expr1-A` from Codex inside the current sandbox failed before any useful benchmark output.
- Error:
```text
OMP: Error #179: Function Can't open SHM2 failed:
OMP: System error #1: Operation not permitted
```
- Interpretation:
  - this is an environment/sandbox shared-memory restriction
  - not a logic bug in the experiment code
  - it affected both:
    - `--seed-workers 5`
    - and `--seed-workers 1`
  - so the issue is broader than just multi-process seed parallelism
- Recommended next step after restart:
  - rerun the same command outside sandbox / with escalated permissions
  - likely command:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_AFFINITY=none \
PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
-m rl_train.benchmarks.staged_steane_experiments \
--stage-specs-json code/rl_train/benchmarks/examples/stage_specs_expr1_phaseA_gate_quick.json \
--stages all \
--seed-workers 1 \
--output-dir code/data_generated/rl_Steane_tune_experiment/expr1_gate_only/phaseA_quick
```

### Current Working Tree Snapshot
- Modified:
  - `code/rl_train/benchmarks/eval_steane_ppo.py`
  - `human_codex_QA.md`
  - `resources/reading_mtrl_/rl_qec_channel_abstract_tikz.pdf`
  - `resources/reading_mtrl_/rl_qec_channel_abstract_tikz.tex`
- Untracked:
  - `code/data_generated/rl_Steane_tune_experiment/README.md`
  - `code/rl_train/benchmarks/eval_steane_cycle_sweep.py`
  - `code/rl_train/benchmarks/examples/stage_specs_expr1_phaseA_gate_quick.json`

### Practical Next Step After Restart
1. Re-run `Expr1-A` outside sandbox.
2. If it completes, inspect:
   - `code/data_generated/rl_Steane_tune_experiment/expr1_gate_only/phaseA_quick/summary.json`
3. Shortlist gate-only conditions for `Expr1-B`.
4. Then prepare/run `Expr2-A`.
