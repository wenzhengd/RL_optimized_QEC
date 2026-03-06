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
1. **先增强统计功效（不改算法**

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
