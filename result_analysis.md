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

## Gap vs Google-Style Workflow and What Must Improve

This section compares our current setup with a Google-style RL-QEC workflow at a practical level.
It is a "borrowed workflow" comparison, not a strict paper reproduction claim.

### 1) Data scale is still too small for stable effect estimation

Current stage4 per-seed training budget:

- `train_calls * n_rounds * shots = 512 * 4 * 4 = 8192` train units/seed.

Our own paper-inspired preset in code (`google_paper_ppo_preset`) corresponds to:

- `8000 * 25 * 400 = 80,000,000` train units/seed.

Implication:

- stage4 is still about `9,765x` smaller than this paper-style preset scale.
- high variance in seed outcomes is therefore expected.

What to change:

- keep PPO, but increase data budget by at least one to two orders of magnitude first
  (before concluding RL effectiveness/non-effectiveness).

### 2) Training signal fidelity is limited by fast proxy mode

Current fast path (`collect_traces=false`) sets:

- `prep_ok_rate = -1`,
- stabilizer means to zeros,
- detector-rate reward signal from `1 - success_rate` proxy.

Implication:

- reward/metric channels become partially collapsed into success proxy,
  reducing gradient quality and making DR/LER evidence less independent.

What to change:

- use a two-phase regime:
  1. fast proxy pretraining (cheap),
  2. short high-fidelity finetuning/evaluation with trace-derived detector metrics.

### 3) Model architecture is minimal relative to temporal syndrome structure

Current policy/value model:

- 2-layer MLP (`hidden_dim=128`) with global Gaussian log-std.

Likely limitation:

- weak handling of temporal/context structure in syndrome evolution and drift.

What to change:

- first upgrade to stronger MLP baseline (`hidden_dim` up, LayerNorm/residual MLP),
- then test recurrent/sequence policy (for example GRU-based actor/critic) for
  round-history-aware control.

### 4) Control protocol is mostly one-shot candidate selection

Current default staged runs:

- `stepping_mode=candidate_eval`, `max_steps=1`
- one action vector is held fixed for the full candidate evaluation window.

Implication:

- policy cannot perform rich round-by-round closed-loop adaptation.

What to change:

- add/expand `online_rounds` experiments (`max_steps=n_rounds`) so policy can
  condition on intermediate outcomes and adapt actions across rounds.

### 5) Statistical protocol still underpowered for publication-grade claim

Current stage4:

- only 5 seeds, large std, one clear regression seed.

What to change:

- power stage with >=15-20 seeds,
- larger post-eval budget (episodes and eval shots),
- pre-registered success criterion + confidence intervals.

### Priority Order (Recommended)

1. Increase seeds and eval budget first (cheapest path to stronger inference).
2. Add high-fidelity detector-metric phase (fix reward/metric fidelity).
3. Scale training data by 10x-100x with seed-level parallelism.
4. Upgrade architecture (stronger MLP -> recurrent/sequence policy).
5. Expand to online round-wise control for adaptive policy behavior.

用中文： (Human write // plz don't delete the following strategy in Chinese. )
  1. **先增强统计功效（不改算法）**

  - 目标：先判断“到底有没有稳定正信号”。
  - 做法：固定当前训练配置，只把 seeds 提到 15-20，并把 post_eval_episodes、eval_shots 提高。
  - 产出：更可靠的 CI / 方差判断。

  2. **再增强训练信号质量（小改环境）**

  - 目标：减少 success_proxy 的信息损失。
  - 做法：保留 fast 训练，但增加“少量 trace-based finetune/eval”阶段。
  - 产出：reward 与最终 metric 更一致，降低误导梯度。

  3. **然后再放大训练数据规模（10x-100x）**

  - 目标：在前两步确认值得后，再烧算力。
  - 做法：沿用现有并行架构扩大 timesteps/shots，做规模曲线。
  - 产出：improvement-vs-cost 曲线，可写进论文。

  4. **最后再做模型结构升级**

  - 目标：如果以上仍受限，再上 GRU/更强 actor-critic。
  - 原因：结构改动大，先把实验基线打稳更划算。

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

## Update (2026-03-06): Stage5 Statistics and Stage6 Trace Finetune

New runs:

- `code/data_generated/steane_staged_runs_power_stats/summary.json` (stage5, 20 seeds)
- `code/data_generated/steane_staged_runs_trace_finetune/summary.json` (stage6, 12 seeds)

Stage5 (`power_stats`, no trace finetune):

- improve(LER~) mean/std: `+5.20% +- 13.70%`
- learned success mean/std: `93.42% +- 1.10%`
- sign count: `13 positive / 6 negative / 1 zero` (n=20)
- one-sided sign-test p-value: `0.1316`

Stage6 (`trace_finetune`, fast phase-1 + small trace phase-2):

- improve(LER~) mean/std: `+12.21% +- 16.46%`
- learned success mean/std: `93.40% +- 1.11%`
- sign count: `10 positive / 2 negative / 0 zero` (n=12)
- one-sided sign-test p-value: `0.0193`

Interim interpretation:

- adding a small trace-based finetune phase improved average relative LER reduction
  in this first comparison (`+12.21%` vs `+5.20%`),
- but stage6 has fewer seeds than stage5, so this should be treated as a promising
  signal, not a final claim.

## Update (2026-03-06): Stage7 Progressive Scale (skip stage6 n=20 extension)

New run:

- `code/data_generated/steane_staged_runs_progressive_scale/summary.json` (stage7, 10 seeds)

Stage7 config intent:

- keep PPO + trace-finetune workflow from stage6,
- moderately increase train/data budget (more rounds, shots, timesteps),
- keep seed-level process parallelism (`--seed-workers 5`) for practical wall-clock.

Stage7 (`progressive_scale`) aggregate:

- improve(LER~) mean/std: `+33.97% +- 9.24%`
- learned success mean/std: `91.72% +- 1.06%`
- sign count: `10 positive / 0 negative` (n=10)
- one-sided sign-test p-value: `0.00098`

Stage7 per-seed LER~ improvement (%):

- seed120 `+48.30`, seed121 `+30.63`, seed122 `+51.68`, seed123 `+21.25`, seed124 `+29.12`
- seed125 `+35.56`, seed126 `+24.15`, seed127 `+35.35`, seed128 `+35.63`, seed129 `+28.02`

Budget comparison (planning units, same accounting as above):

- stage6 per-seed units: `35,840`
- stage7 per-seed units: `89,088` (`2.49x` vs stage6)
- stage6 total units: `430,080` (12 seeds)
- stage7 total units: `890,880` (10 seeds, `2.07x` vs stage6 total)

Interim interpretation:

- this progressive scale step produced a clear and consistent positive signal
  across all seeds in this run,
- compared with stage6, effect size increased (`+33.97%` vs `+12.21%`) and variance
  shrank (`9.24%` vs `16.46%` std),
- this is strong internal evidence that increased data budget and trace-finetune
  are helping in our Steane pipeline.

## Update (2026-03-06): Stage8-10 Scale Curve (progressive high-budget sweep)

New run:

- `code/data_generated/steane_staged_runs_scale_curve/summary.json`
  (stage8/9/10 in one continuous run)

### Aggregate results

| Stage | Seeds | improve(LER~) mean +- std | 95% CI (mean) | learned success mean +- std | sign count | one-sided sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| stage8_scale_x3 | 10 | `+49.37% +- 12.66%` | `[+41.52%, +57.22%]` | `93.87% +- 0.55%` | `10+/0-/0` | `0.00098` |
| stage9_scale_x10 | 8 | `+48.96% +- 6.41%` | `[+44.52%, +53.40%]` | `93.88% +- 0.44%` | `8+/0-/0` | `0.00391` |
| stage10_scale_x30 | 6 | `+49.30% +- 11.79%` | `[+39.87%, +58.73%]` | `94.05% +- 0.20%` | `6+/0-/0` | `0.01563` |

### Compute-cost scaling (planning units)

Using the same unit definition as above:

| Stage | total units | vs stage6 total | vs previous stage |
|---|---:|---:|---:|
| stage6_trace_finetune | `430,080` | `1.00x` | - |
| stage7_progressive_scale | `890,880` | `2.07x` | `2.07x` |
| stage8_scale_x3 | `2,227,200` | `5.18x` | `2.50x` |
| stage9_scale_x10 | `5,606,400` | `13.04x` | `2.52x` |
| stage10_scale_x30 | `13,307,904` | `30.94x` | `2.37x` |

### Interpretation for Step-3 objective

- Physical improvement is now consistently positive in all seeds for stage7-10.
- The effect grows strongly from stage6 to stage8 (`+12.21% -> +49.37%`), then
  enters a plateau around `~49%` for stage8/9/10.
- This means Step-3 ("scale data budget and build improvement-vs-cost curve") is
  effectively achieved up to about `31x` total compute vs stage6.
- Marginal gain after stage8 is small under the current PPO + environment setup,
  which suggests near-saturation for this configuration (not necessarily global optimum).

Practical conclusion:

- We now have strong internal evidence that increasing data budget (with the
  current trace-finetune workflow) materially improves Steane performance.
- For next gains, algorithm/model/protocol changes are likely to matter more than
  simply pushing the same pipeline to much higher compute.

## Update (2026-03-06): Stage11 Architecture Fairness (Wider LayerNorm MLP)

New run:

- `code/data_generated/steane_staged_runs_arch_mlp/summary.json` (stage11, 10 seeds)

Protocol:

- stage11 keeps stage8 budget/protocol fixed (same timesteps/shots/rounds/eval),
- only architecture changes:
  - `ppo_hidden_dim: 256` (from 128),
  - `ppo_use_layer_norm: true` (stage8 was false).

### Stage8 vs Stage11 (equal-budget comparison)

| Stage | Seeds | improve(LER~) mean +- std | 95% CI (mean) | learned success mean +- std | sign count | one-sided sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| stage8_scale_x3 (baseline) | 10 | `+49.37% +- 12.66%` | `[+41.52%, +57.22%]` | `93.87% +- 0.55%` | `10+/0-/0` | `0.00098` |
| stage11_arch_mlp (wider+LN) | 10 | `+33.86% +- 13.99%` | `[+25.19%, +42.53%]` | `92.16% +- 1.40%` | `10+/0-/0` | `0.00098` |

Delta (stage11 - stage8):

- improve(LER~): `-15.51` percentage points
- learned success: `-1.71` percentage points

Interim interpretation:

- under this equal-budget setup, wider LayerNorm MLP is still beneficial vs fixed-zero
  baseline (all seeds positive), but underperforms the stage8 baseline architecture.
- likely implication: architecture-only widening/LN is not sufficient here without
  re-tuning PPO hyperparameters (learning rate, entropy coefficient, rollout/update mix).

## Update (2026-03-06): Step-4 Option 1 (PPO hyperparameter retune for wider+LN MLP)

New runs:

- `code/data_generated/steane_staged_runs_arch_tune/summary.json` (stage12/13/14, tuning sweep)
- `code/data_generated/steane_staged_runs_arch_tuned_confirm/summary.json` (stage15, 10-seed confirmation)

### Tuning sweep (same budget as stage8/stage11, 5 seeds each)

| Stage | Key tuning change | improve(LER~) mean +- std | learned success mean +- std |
|---|---|---:|---:|
| stage12_arch_mlp_tune_a | lower LR (`1e-4`) + trace LR (`5e-5`), entropy `0.01` | `+50.42% +- 9.45%` | `93.30% +- 1.24%` |
| stage13_arch_mlp_tune_b | lower LR + lower entropy (`0.005`) | `+42.74% +- 6.21%` | `93.41% +- 0.31%` |
| stage14_arch_mlp_tune_c | moderate LR + more PPO updates (`epochs=6`) | `+39.04% +- 7.64%` | `93.49% +- 1.13%` |

Selection:

- `stage12 (tune_a)` selected as best candidate for confirmation.

### Confirmation run (stage15, 10 seeds)

| Stage | Seeds | improve(LER~) mean +- std | 95% CI (mean) | learned success mean +- std | sign count | one-sided sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| stage11_arch_mlp (untuned) | 10 | `+33.86% +- 13.99%` | `[+25.19%, +42.53%]` | `92.16% +- 1.40%` | `10+/0-/0` | `0.00098` |
| stage15_arch_mlp_tuned_confirm | 10 | `+46.55% +- 8.33%` | `[+41.38%, +51.71%]` | `93.49% +- 0.72%` | `10+/0-/0` | `0.00098` |
| stage8_scale_x3 baseline | 10 | `+49.37% +- 12.66%` | `[+41.52%, +57.22%]` | `93.87% +- 0.55%` | `10+/0-/0` | `0.00098` |

Delta summary:

- stage15 - stage11: `+12.69` percentage points (LER~ improvement)
- stage15 - stage8: `-2.82` percentage points (LER~ improvement)

Interpretation:

- Hyperparameter retuning substantially recovered architecture performance
  (`stage11 -> stage15`), validating that the initial stage11 drop was largely
  optimization mismatch, not a hard architecture failure.
- After tuning, wider+LayerNorm MLP is close to stage8 baseline but still not
  clearly better in this 10-seed confirmation.

## Update (2026-03-07): Stage7 with Composed Correlated Channel (`f=1e3`, `g=1.6`)

New run:

- `code/data_generated/steane_stage7_composed_f1e3_g16/summary.json`
  (stage7 only, 10 seeds)

Protocol notes:

- Stage template: default `stage7_progressive_scale`
- Global channel override: `composed_google_gate_specific_correlated`
- Correlated params: `steane_channel_corr_f=1000.0`, `steane_channel_corr_g=1.6`,
  `steane_channel_corr_g_mode=per_circuit`
- Seed parallelism: `--seed-workers 5`

### Aggregate result

| Stage | Seeds | improve(LER~) mean +- std | 95% CI (mean) | learned success mean +- std | sign count | one-sided sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| stage7_progressive_scale (composed f1e3 g1.6) | 10 | `+36.91% +- 6.96%` | `[+32.60%, +41.22%]` | `91.63% +- 1.60%` | `10+/0-/0` | `0.00098` |

### Comparison against prior stage7 record

- Prior stage7 in this document: `+33.97% +- 9.24%` (LER~ improvement).
- Current composed-correlated stage7: `+36.91% +- 6.96%`.
- Delta (current - prior): `+2.94` percentage points.

Interpretation:

- Under this composed correlated setting, stage7 again shows a strong and
  consistent positive learning effect (all 10 seeds positive).
- Variance is lower than the prior stage7 record, which improves confidence in
  the sign and stability of the effect for this setting.
- This comparison is indicative (not strictly apples-to-apples causal), because
  the noise-channel family/parameters differ from the earlier stage7 baseline.

## Update (2026-03-07): Stage8 with Composed Correlated Channel (`f=1e3`, `g=1.6`)

New run:

- `code/data_generated/steane_stage8_composed_f1e3_g16/summary.json`
  (stage8 only, 10 seeds)

Protocol notes:

- Stage template: default `stage8_scale_x3`
- Global channel override: `composed_google_gate_specific_correlated`
- Correlated params: `steane_channel_corr_f=1000.0`, `steane_channel_corr_g=1.6`,
  `steane_channel_corr_g_mode=per_circuit`
- Seed parallelism: `--seed-workers 5`

### Aggregate result

| Stage | Seeds | improve(LER~) mean +- std | 95% CI (mean) | learned success mean +- std | sign count | one-sided sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| stage8_scale_x3 (composed f1e3 g1.6) | 10 | `+40.64% +- 15.75%` | `[+30.87%, +50.40%]` | `92.67% +- 0.85%` | `10+/0-/0` | `0.00098` |

### Comparison against current composed stage7 run

- stage7 (composed f1e3 g1.6): `+36.91% +- 6.96%`
- stage8 (composed f1e3 g1.6): `+40.64% +- 15.75%`
- Delta (stage8 - stage7): `+3.73` percentage points.

Interpretation:

- Stage8 remains strongly positive under the same composed correlated setting
  (all 10 seeds positive), reinforcing the learning-advantage claim.
- Mean improvement is higher than stage7 in this run, but variance is also
  larger; this suggests the gain is real but still sensitive to seed-level
  stochasticity at this budget.
- Next step is stage9 under the same settings to test whether the curve
  continues improving or enters a clearer plateau region.
