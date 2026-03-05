# RL Improves Steane Code: Interim Result Analysis

Date: 2026-03-05

## Scope

This document summarizes current PPO-on-Steane staged experiments, with focus on:

- performance evidence for "RL improves Steane code",
- compute/resource cost scaling across stages,
- publication-readiness assessment and next steps.

## Data Sources

- `code/data_generated/steane_staged_runs/summary.json` (stage1-3)
- `code/data_generated/steane_staged_runs_power/summary.json` (stage4)
- per-seed JSON files under:
  - `code/data_generated/steane_staged_runs/stage*/`
  - `code/data_generated/steane_staged_runs_power/stage4_power/`

All numbers below are extracted from these files.

## Protocol Snapshot

Common settings:

- optimizer: PPO
- backend: Steane adapter
- control mode: gate-specific
- stepping mode: candidate_eval
- reward mode: paper_surrogate
- traces: disabled (`collect_traces=false`, fast summary path)

Stage configs:

| Stage | Seeds | total_timesteps | rollout_steps | n_rounds | train shots/step | post_eval_episodes | eval shots/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage1_sanity | 1 | 20 | 2 | 1 | 1 | 2 | 1 |
| stage2_pilot | 2 | 60 | 10 | 1 | 2 | 2 | 2 |
| stage3_scale | 3 | 210 | 16 | 2 | 2 | 2 | 2 |
| stage4_power | 5 | 512 | 32 | 4 | 4 | 8 | 24 |

Note: stage4 uses larger evaluation shots only (`eval_steane_shots_per_step=24`) to improve metric stability without multiplying training cost by the same factor.

## Resource Cost Accounting

Cost unit (for planning) is defined as:

`total_units = seeds * (train_calls * n_rounds * train_shots + eval_calls * n_rounds * eval_shots)`

where:

- `train_calls = floor(total_timesteps / rollout_steps) * rollout_steps`
- `eval_calls = 3 * post_eval_episodes` (learned/fixed/random)

Computed units:

| Stage | train_calls/seed | eval_calls/seed | train_units | eval_units | total_units | ratio vs previous |
|---|---:|---:|---:|---:|---:|---:|
| stage1_sanity | 20 | 6 | 20 | 6 | 26 | - |
| stage2_pilot | 60 | 6 | 240 | 24 | 264 | 10.15x |
| stage3_scale | 208 | 6 | 2496 | 72 | 2568 | 9.73x |
| stage4_power | 512 | 24 | 40960 | 11520 | 52480 | 20.44x |

Interpretation:

- stage1->2->3 follows the intended ~10x scaling.
- stage4 intentionally adds a larger jump (~20x vs stage3) for statistical power.

## Performance Results

Stage-level aggregates (vs fixed-zero baseline):

| Stage | improve(LER~) mean +- std | improve(DR) mean +- std | learned success mean +- std |
|---|---:|---:|---:|
| stage1_sanity | +0.00% +- 0.00% | +0.00% +- 0.00% | 100.00% +- 0.00% |
| stage2_pilot | +0.00% +- 0.00% | +0.00% +- 0.00% | 100.00% +- 0.00% |
| stage3_scale | +33.33% +- 47.14% | +33.33% +- 47.14% | 100.00% +- 0.00% |
| stage4_power | +31.63% +- 51.92% | +31.63% +- 51.92% | 95.62% +- 1.17% |

Stage4 seed-level (LER proxy improvement vs fixed-zero):

| Seed | learned success | fixed-zero success | random success | improve(LER~) |
|---:|---:|---:|---:|---:|
| 50 | 95.83% | 90.10% | 79.69% | +57.89% |
| 51 | 95.31% | 90.63% | 79.69% | +50.00% |
| 52 | 97.40% | 91.67% | 89.58% | +68.75% |
| 53 | 93.75% | 96.35% | 86.46% | -71.43% |
| 54 | 95.83% | 91.15% | 75.00% | +52.94% |

Additional stage4 summary:

- mean absolute success gain (learned - fixed): `+3.65` percentage points
- std of absolute success gain: `3.16` percentage points
- sign count: `4/5` seeds positive, `1/5` negative
- one-sided sign-test p-value for positive effect: `0.1875`

## Runtime Notes (Observed)

From seed JSON modification times:

- stage1-3 completed quickly (minutes scale).
- stage4 completion stamps:
  - seed50: 14:43:14
  - seed51: 15:13:48
  - seed52: 15:29:49
  - seed53: 15:45:54
  - seed54: 16:02:03

Empirically, stage4 seeds were mostly ~16 minutes/seed after the first, and total wall-clock for the stage4 run was roughly ~1.7 hours in this environment.

## Interpretation

Current evidence level:

- There is a positive trend in stage4 (4/5 seeds beat fixed-zero).
- Variance is still high (large std, one clear regression seed).
- Statistical confidence is not yet strong enough for a strong publication claim.

Important caveat:

- with `collect_traces=false`, detector-rate metric in current fast path is a success-derived proxy, so DR and LER~ are not fully independent evidence channels.

## Publication Readiness Assessment

Supported today:

- end-to-end PPO training/evaluation pipeline works on Steane adapter,
- nontrivial positive signal appears under higher-budget stage4 settings.

Not yet supported:

- robust, low-variance, statistically significant claim that RL improves Steane performance.

## Recommended Next Experiments

1. Increase seeds for power stage (target at least 15-20 seeds).
2. Keep training budget fixed, but increase evaluation budget further:
   - larger `post_eval_episodes`,
   - larger `eval_steane_shots_per_step` (for tighter confidence intervals).
3. Add a final "trace-enabled evaluation only" pass for top checkpoints to report detector/stabilizer metrics beyond success proxy.
4. Report confidence intervals for absolute success gain and failure-rate reduction.
5. Pre-register a clear success criterion (for example: mean failure-rate reduction > X% with CI excluding zero).

