# Issue: Correlated Channel Simulation Performance

## Status
- Open
- Priority: Medium-High
- Scope: `correlated_pauli_noise_channel` runtime in Steane RL loops
- Progress:
  - Phase A complete: correlated kernel vectorization + safe event caching
  - Phase B partial complete: one-pass composed channel executor added for
    `google + correlated` composition

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

## Risk
- Larger staged runs can become much longer when switched to correlated channel.
- This can reduce experiment throughput and increase iteration cost.

## Why This Is Correct (and not a bug)
- The correlated model must preserve hidden-state evolution across idle windows and reset only at shot boundaries.
- Reusing a previously generated noisy circuit would break Markov-state semantics and produce incorrect physics/statistics.

## Candidate Optimizations (Future Work)
1. Batch append operations to Stim where possible to reduce Python call overhead.
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
