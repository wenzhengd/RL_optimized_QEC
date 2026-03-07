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
