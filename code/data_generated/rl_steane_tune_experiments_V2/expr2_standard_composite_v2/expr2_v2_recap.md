# Expr2 V2 Recap

This note summarizes the current `Expr2 V2` status after the stronger-training
`Phase A` and `Phase B focused` reruns.

## Setup Decisions That Are Now Fixed

- use gate slice `1q/2q = 1/10`
- use balanced composite settings over:
  - `scale in {0.02, 0.025}`
  - `f in {1e2, 1e3, 1e4}`
  - `g in {0.4, 1.0, 1.6}`
- fully inherit the stronger `Expr1 V2` training recipe:
  - `total_timesteps = 4096`
  - `steane_shots_per_step = 16`
  - `rollout_steps = 64`
  - `ppo_learning_rate = 1e-4`
  - `ppo_ent_coef = 0.001`
  - `steane_action_penalty_coef = 0.005`
  - `steane_miscal_penalty_coef = 0.001`

## Phase B Focused Results

Source:

- `phaseB_focused/summary.json`

Focused shortlist that was tested:

- `scale=0.02, f=1e2, g=0.4`
- `scale=0.025, f=1e3, g=1.6`
- `scale=0.025, f=1e4, g=1.0`
- `scale=0.02, f=1e3, g=0.4`

Observed aggregate results:

- `scale=0.025, f=1e3, g=1.6`
  - `improve(LER~) = +0.3064 +- 0.1520`
  - `learned.success_rate = 0.9808`
  - positive-gain seeds: `9/10`
- `scale=0.025, f=1e4, g=1.0`
  - `improve(LER~) = +0.2272 +- 0.1106`
  - `learned.success_rate = 0.9811`
  - positive-gain seeds: `10/10`
- `scale=0.02, f=1e2, g=0.4`
  - `improve(LER~) = +0.2242 +- 0.1454`
  - `learned.success_rate = 0.9854`
  - positive-gain seeds: `9/10`
- `scale=0.02, f=1e3, g=0.4`
  - `improve(LER~) = +0.0696 +- 0.2305`
  - `learned.success_rate = 0.9850`
  - positive-gain seeds: `5/10`

## Current Interpretation

Three conditions are now strong enough to treat as main `Expr2 V2` candidates:

- `scale=0.025, f=1e3, g=1.6`
- `scale=0.025, f=1e4, g=1.0`
- `scale=0.02, f=1e2, g=0.4`

These are the best current `Expr2 V2` headline points because they combine:

- clearly positive mean gain
- acceptable dispersion
- strong seed-level positive-gain ratio

The remaining focused boundary point

- `scale=0.02, f=1e3, g=0.4`

should not be promoted as a headline benchmark condition. It is better treated
as a boundary / comparison point because the mean gain is small and the seed
stability is weak.

## Recommended Confirm Stage

`Phase C confirm` should only carry the three main candidates:

- `scale=0.025, f=1e3, g=1.6`
- `scale=0.025, f=1e4, g=1.0`
- `scale=0.02, f=1e2, g=0.4`

Recommended purpose:

- verify that the `Phase B focused` ranking survives another higher-evidence run
- confirm that the top candidate is not a seed artifact
- produce the final `Expr2 V2` benchmark table for writeup
