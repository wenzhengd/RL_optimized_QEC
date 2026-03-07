# Issue: Correlated Channel Simulation Performance

## Status
- Open
- Priority: Medium-High
- Scope: `correlated_pauli_noise_channel` runtime in Steane RL loops

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

## Risk
- Larger staged runs can become much longer when switched to correlated channel.
- This can reduce experiment throughput and increase iteration cost.

## Why This Is Correct (and not a bug)
- The correlated model must preserve hidden-state evolution across idle windows and reset only at shot boundaries.
- Reusing a previously generated noisy circuit would break Markov-state semantics and produce incorrect physics/statistics.

## Candidate Optimizations (Future Work)
1. Vectorize correlated idle-window sampling over qubits/axes (replace Python per-qubit loops with NumPy batch ops).
2. Cache timeline/event expansion safely (cache deterministic structure only, not sampled noisy outputs).
3. Batch append operations to Stim where possible to reduce Python call overhead.
4. Add optional profiling hooks to report `apply` time breakdown by model and sub-circuit.
5. Revisit parallel strategy for correlated channel only if semantic guarantees can be preserved.

## Suggested Acceptance Criteria For A Future Fix
1. Preserve correlated-channel semantics (same shot-level statefulness behavior).
2. Keep deterministic reproducibility behavior under fixed seeds.
3. Achieve at least `1.5x` speedup for correlated channel on representative staged/eval workloads.
4. Add a regression benchmark script and record baseline vs improved timings.
