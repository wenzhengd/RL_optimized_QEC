# RL Steane Tuning Experiments V2

This folder is the reset point for the next round of Steane-RL benchmark work.

V2 is not a direct continuation of V1. The main reason is that V1 likely mixed
reasonable gate-level parameter choices with unreasonable circuit-level noise
budgets, especially for deep memory/QEC circuits. As a result, some conclusions
from V1 may be confounded by over-heavy noise regimes rather than reflecting
the true quality of the RL controller.

The immediate goal of V2 is therefore:

1. recalibrate parameter regimes at the circuit level
2. rerun experiments under better-controlled noise budgets
3. advance one experiment at a time instead of sweeping the whole family in parallel

See V1 reference material:

- [README.md](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_Steane_tune_experiment/README.md)
- [experiment_summary.md](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_Steane_tune_experiment/artifacts/experiment_summary.md)
- [human_codex_QA.md](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/human_codex_QA.md)

## Why V2 Exists

The recent review of `Expr1` surfaced several problems:

- `learned_success` is below `fixed_zero_success` in part of the gate-only sweep.
- `fixed_zero` is only a zero-action baseline, not a guaranteed-safe or optimal baseline.
- `overall gate-noise scale` in V1 was chosen as a gate-level multiplier, but not clearly calibrated against the total circuit depth / total fault budget.
- In `1q/2q = 1/4`, the effective 2q depolarizing probability can become heavy enough that the whole memory experiment may already be in a multi-fault-dominated regime.

This creates a real risk that some V1 sweeps answered:

- "what happens under over-heavy noise?"

instead of the intended question:

- "when does RL meaningfully improve QEC control under realistic and comparable regimes?"

## V2 Principles

### 1. Circuit-Level Calibration Comes First

Do not choose sweep values only from raw channel knobs such as:

- `channel_regime_a`
- `channel_regime_b`
- correlated `f`
- correlated `g`
- measurement flip probability

Before using them as benchmark axes, map them to circuit-level difficulty.

Recommended calibration quantities:

- expected gate-fault count per candidate-eval circuit
- approximate no-fault probability per circuit
- baseline `fixed_zero.success_rate`
- baseline `fixed_zero.ler_proxy`

The target is to avoid both extremes:

- too light: baseline success is so high that RL gain is numerically tiny
- too heavy: the circuit is already saturated by faults, so RL signal is washed out

### 2. Use `fixed_zero` Correctly

In this project, `fixed_zero` means:

- zero-action control policy

It does **not** mean:

- physically optimal no-RL policy
- guaranteed lower bound on noise
- "original hardware behavior" in any strict sense

So negative RL gain is possible and must be treated as a real failure case, not
as an impossible artifact.

Practical interpretation for `Expr1 V2`:

- if `learned.success_rate < fixed_zero.success_rate`, the result should be read as:
  - PPO did not find a control vector better than the trivial zero-action guess
  - under the current observation, reward, and training budget
- this does not mean the noise model is inconsistent
- it means the learned policy is effectively farther from the useful hidden control than `u = 0` is, on the gates/slots that matter for logical success
- in other words, `learned < fixed_zero` is a training-and-inference failure case, not a physics failure case

### 3. Prefer Absolute and Relative Metrics Together

Do not report only relative improvement.

Primary reporting set for V2:

- `learned.success_rate`
- `fixed_zero.success_rate`
- `delta_success = learned.success_rate - fixed_zero.success_rate`
- `improvement_vs_fixed_zero.ler_proxy`

Secondary reporting:

- seed-level positive-gain ratio
- mean +- std
- circuit-level calibration quantities

### 4. Advance One Experiment at a Time

V2 should not start by relaunching all experiments.

Recommended order:

1. calibrate and rerun `Expr1` only
2. use the `Expr1` calibration lessons to redesign `Expr2`
3. only then revisit `Expr3`
4. leave `Expr4` for last unless it depends directly on recalibrated upstream winners

This is the preferred strategy because:

- `Expr1` is the simplest place to validate regime selection
- if `Expr1` is miscalibrated, the same mistake may propagate into `Expr2/3/4`
- it is cheaper to fail early on one experiment than to invalidate a whole benchmark suite

## Proposed V2 Workflow

### Phase 0: Regime Audit

For each existing V1 experiment family:

- list the active noise knobs
- estimate the corresponding circuit-level effective noise budget
- label each condition as:
  - `too_light`
  - `usable`
  - `too_heavy`

Deliverable:

- `regime_audit.md`

### Phase 1: Expr1 Recalibration

Start with gate-only noise because it is the cleanest case.

Tasks:

- use an explicitly low-to-mid `overall gate-noise scale` grid
- consider replacing the raw scale axis with a derived effective-noise quantity
- choose a regime window where `fixed_zero` is neither near-perfect nor already collapsed
- rerun only the recalibrated `Expr1`

Deliverables:

- `expr1_gate_only_v2/`
- `expr1_recalibration_notes.md`

### Expr1 V2 Working Grid

Current agreed working grid for `Expr1 V2`:

- `overall gate-noise scale in {0.01, 0.025, 0.05, 0.1, 0.25, 0.5}`
- ratio slice A: `1q/2q = 1/1`
- ratio slice B: `1q/2q = 1/10`

Mapped to regime parameters:

- `1q/2q = 1/1`:
  - `channel_regime_a = s`
  - `channel_regime_b = s`
- `1q/2q = 1/10`:
  - `channel_regime_a = s`
  - `channel_regime_b = 10 * s`

Important note:

- this is the current V2 working grid, not yet a validated final regime
- it should still be checked against circuit-level calibration signals such as:
  - `fixed_zero.success_rate`
  - approximate no-fault probability
  - expected gate-fault count per candidate-eval circuit
- if the high end (`0.25`, `0.5`) still looks too heavy, V2 should trim it after the first calibration pass

### Expr1 V2 Training Levers

If the goal is to increase the chance that `learned > fixed_zero` without
changing the noise parameters themselves, prefer improving training signal
quality and optimization stability before changing the regime grid.

Recommended levers:

- increase `total_timesteps`
  - current quick-scan budgets are useful for screening, but are too small to guarantee that PPO beats the zero-action baseline in a 48-dimensional action space
- increase `steane_shots_per_step`
  - this reduces reward variance and makes it harder for PPO to overfit Monte Carlo noise
- increase `rollout_steps`
  - larger PPO batches usually stabilize updates in this setting
- reduce optimizer aggressiveness
  - lower `ppo_learning_rate`
  - reduce overly strong entropy pressure when exploration keeps producing unnecessary nonzero actions
- strengthen regularization toward safe actions
  - slightly larger `steane_action_penalty_coef`
  - add a nonzero miscalibration penalty when the aim is to avoid policies that are worse than `fixed_zero`
- prefer trace-finetune or trace-eval for focused and confirm stages
  - the fast path is fine for screening, but the final selection should rely on cleaner evaluation signal
- compare candidate settings by multi-seed mean and positive-seed ratio, not by a single seed

Concrete first-pass recipe for moving a quick `Expr1 V2` run toward a setting
where `learned > fixed_zero` is more likely:

- `total_timesteps: 512 -> 4096`
- `steane_shots_per_step: 4 -> 16`
- `rollout_steps: 32 -> 64`
- `ppo_learning_rate: 3e-4 -> 1e-4`
- `ppo_ent_coef: 0.01 -> 0.001`
- `steane_action_penalty_coef: 0.001 -> 0.005` or `0.01`
- set a nonzero `steane_miscal_penalty_coef`

Rationale for this package:

- more timesteps and more shots reduce the chance that PPO simply overfits reward noise
- larger rollouts make updates less brittle
- lower learning rate and lower entropy make it easier for the policy to stop making unnecessary nonzero moves
- stronger action and miscalibration penalties make `u = 0` a stronger fallback unless the data supports moving away from it

Default interpretation guideline:

- `Phase A` can establish whether a regime is learnable
- `Phase B/C` should establish whether RL robustly beats `fixed_zero`
- if a condition only beats `fixed_zero` at one seed or one very small budget, it should not be promoted as a strong RL-positive result

### Phase 2: Expr2 Redesign

Only after `Expr1` looks physically sensible:

- revisit the `(f, g)` grid for composite/correlated noise
- check whether V1 values such as `g in {0.4, 1.0, 1.6}` remain sensible after circuit-level calibration
- define a smaller pilot grid first

Deliverables:

- `expr2_standard_composite_v2/`
- `expr2_grid_design.md`

### Expr2 V2 Guidance: Balance Gate and Correlated Strength

For `Expr2 V2`, the main new design rule should be:

- avoid choosing gate-noise and correlated-idle-noise settings that are badly imbalanced in average circuit-level strength

Target principle:

- the mean gate-noise contribution and the mean correlated-idle contribution
  should be in the same order of magnitude
- ideally within a small factor of each other
- at minimum, avoid combinations where one side dominates the other by much
  more than about `10x`

Why this matters:

- if gate noise is much stronger than correlated idle noise, `Expr2` becomes
  close to "mostly gate-noise experiment with a weak idle perturbation"
- if correlated idle noise is much stronger than gate noise, `Expr2` stops being
  a balanced composite benchmark and becomes mostly an idle-channel benchmark
- either failure mode makes it harder to interpret what the RL controller is
  actually learning

Practical comparison rule:

- compare both channel families at the level of one `candidate_eval` circuit
- do not compare one side per gate and the other side per circuit without
  converting units

Approximate quantities used in the current stack:

- gate channel:
  - use expected gate-fault budget per candidate-eval circuit
  - this depends on:
    - `channel_regime_a`
    - `channel_regime_b`
    - base `p_1q/p_2q`
    - sensitivity terms
    - typical control mismatch
- correlated channel:
  - use the per-circuit correlated idle budget implied by
    `channel_corr_g_mode = per_circuit`
  - in current code this is driven mainly by:
    - `channel_corr_g`
    - `p_1q_base`
    - `sensitivity_1q`
    - typical control mismatch

Important interpretation note:

- `f` mainly changes temporal correlation structure
- `g` mainly changes average correlated-idle strength
- so when checking average-strength matching, the key pair is:
  - gate scale vs `g`
  - not gate scale vs `f`

### Expr2 V2 Recommended Starting Point

Current recommended `Expr2 V2` gate slice:

- fix `1q/2q = 1/10`
- test gate-noise scale in:
  - `{0.02, 0.025, 0.05}`

Current recommended correlated grid inherited from V1:

- `f in {1e2, 1e3, 1e4}`
- `g in {0.4, 1.0, 1.6}`
- `steane_channel_corr_g_mode = per_circuit`

This V1 grid should still be treated as a good starting grid, not as a
physically validated final truth.

### Expr2 V2 Scale-Matching Assessment

Using the current code defaults and a simple representative mismatch estimate,
the gate slice `1q/2q = 1/10` with `scale in {0.02, 0.025, 0.05}` is broadly
compatible with the V1 correlated grid, but not uniformly so.

Approximate assessment:

- `scale = 0.02`
  - compatible with `g = 1.0`
  - compatible with `g = 1.6`
  - acceptable but somewhat gate-heavy with `g = 0.4`
- `scale = 0.025`
  - compatible with `g = 1.0`
  - compatible with `g = 1.6`
  - acceptable but somewhat gate-heavy with `g = 0.4`
- `scale = 0.05`
  - compatible with `g = 1.0`
  - compatible with `g = 1.6`
  - borderline too gate-heavy with `g = 0.4`

So the main caution flag is:

- `scale = 0.05` together with `g = 0.4`

This combination should not be treated as a core balanced composite benchmark
unless additional calibration confirms that the resulting circuit-level budgets
remain interpretable.

### Expr2 V2 Practical Recommendation

If the goal is a conservative and well-balanced first pass, prefer:

- full grid over:
  - `scale in {0.02, 0.025}`
  - `f in {1e2, 1e3, 1e4}`
  - `g in {0.4, 1.0, 1.6}`
- and then add `scale = 0.05` only for:
  - `g in {1.0, 1.6}`

This keeps the broad V1 `(f, g)` coverage while reducing the chance that
`Expr2 V2` is accidentally dominated by the gate part of the composite channel.

If a broader first-pass sweep is still desired, then:

- keep `scale = 0.05, g = 0.4` in the table
- but mark it explicitly as a boundary / imbalance-check point, not as a
  headline benchmark condition

### Expr2 V2 Training Rule: Fully Inherit Expr1 Stronger Settings

For `Expr2 V2`, the training configuration should not fall back to the lighter
quick-scan budget once the V2 design decision has been made.

The practical rule is:

- `Expr2` must fully inherit the stronger `Expr1 V2` training recipe
- do not mix the new `Expr2` regime design with the old light-budget training
  setup from V1-style quick scans

Required training settings for `Expr2 V2` pilot/focused reruns:

- `total_timesteps = 4096`
- `steane_shots_per_step = 16`
- `rollout_steps = 64`
- `ppo_learning_rate = 1e-4`
- `ppo_ent_coef = 0.001`
- `steane_action_penalty_coef = 0.005` or `0.01`
- keep a nonzero `steane_miscal_penalty_coef`

Interpretation rule:

- any earlier `Expr2 V2` run that used the lighter `512 / 32 / 6`-style budget
  should be treated as superseded once the full stronger-training rule is
  adopted
- after this point, the canonical `Expr2 V2` results are only the reruns that
  use the full stronger configuration above

### Expr2 V2 Relation to V1

V1 `Expr2` did use:

- `f in {1e2, 1e3, 1e4}`
- `g in {0.4, 1.0, 1.6}`

and then selected focused follow-up conditions from the numerical results.

However, V1 did not explicitly enforce a "gate vs correlated average strength
must be matched" rule before launching the sweep.

So the intended V2 improvement is:

- keep the useful V1 correlated grid
- but choose the gate slice more carefully
- and explicitly reject badly imbalanced composite settings before promoting
  them into focused or confirm stages

### Expr2 V2 Outcome

`Expr2 V2` is complete through:

- `Phase A pilot`
- `Phase B focused`
- `Phase C confirm`

Headline result:

- `Expr2 V2` does produce stable RL-positive composite settings once the
  experiment fully inherits the stronger `Expr1 V2` training recipe

Confirmed ranking from `Phase C`:

1. `scale=0.025, f=1e4, g=1.0`
   - `improve(LER~) = +0.2070 +- 0.1344`
2. `scale=0.025, f=1e3, g=1.6`
   - `improve(LER~) = +0.1811 +- 0.1925`
3. `scale=0.02, f=1e2, g=0.4`
   - `improve(LER~) = +0.1073 +- 0.1737`

Interpretation:

- the best current `Expr2 V2` headline point is `scale=0.025, f=1e4, g=1.0`
- `scale=0.025, f=1e3, g=1.6` is a close secondary winner
- `scale=0.02, f=1e2, g=0.4` remains RL-positive, but is weaker at confirm level

Recommended GitHub-facing readout:

- best balanced composite point:
  - `scale=0.025, f=1e4, g=1.0`
- shortest conclusion:
  - RL remains effective after adding correlated noise, but only in a subset of
    well-balanced composite regimes

Figures and summaries:

- [Expr2 recap](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr2_standard_composite_v2/expr2_v2_recap.md)
- [Phase A summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr2_standard_composite_v2/phaseA_pilot_balanced/summary.json)
- [Phase B summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr2_standard_composite_v2/phaseB_focused/summary.json)
- [Phase C summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr2_standard_composite_v2/phaseC_confirm/summary.json)
- [Expr2 V2 Phase A figure](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr2_standard_composite_v2/plots/expr2_v2_phaseA_heatmaps.png)
- [Expr2 V2 Phase B/C figure](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr2_standard_composite_v2/plots/expr2_v2_phaseBC_compare.png)

### Expr2 Transfer Check

We also tested whether an `Expr1` gate-only learned control can be transplanted
directly into the confirmed `Expr2` composite conditions.

This transfer batch used `3` source seeds from the `Expr1 V2` gate-only
condition `ratio=1/10, scale=0.025`, and injected them into the top `3`
confirmed `Expr2` conditions.

Key result:

- the `Expr1-trained` transfer policy does **not** reliably explain the
  `Expr2` gains
- across all three confirm targets, `Expr2-trained` remains better than
  transferred `Expr1-trained`
- in two of the three targets, the transfer policy is on average even worse than
  `fixed_zero`

Current `3x3` transfer summary:

1. `scale=0.025, f=1e4, g=1.0`
   - transfer vs `fixed_zero`: `improve(LER~) = -0.0758 +- 0.1553`
   - transfer vs `Expr2-trained`: `delta success = -0.003581 +- 0.000797`
2. `scale=0.025, f=1e3, g=1.6`
   - transfer vs `fixed_zero`: `improve(LER~) = +0.0194 +- 0.2538`
   - transfer vs `Expr2-trained`: `delta success = -0.011176 +- 0.007997`
3. `scale=0.02, f=1e2, g=0.4`
   - transfer vs `fixed_zero`: `improve(LER~) = -0.0955 +- 0.2309`
   - transfer vs `Expr2-trained`: `delta success = -0.006293 +- 0.003609`

Interpretation:

- `Expr2` gains should not be described as simple reuse of an `Expr1`
  gate-only policy
- the safer conclusion is that `Expr2-trained` policies learn additional
  adaptation to the composite environment

Supporting files:

- [Transfer summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/policy_transfer/transfer_summary.md)
- [Transfer success-rate comparison](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/policy_transfer/transfer_compare.png)
- [Transfer delta figure](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/policy_transfer/transfer_deltas.png)

### Phase 3: Expr3 and Expr4

Proceed only after the above two phases are stable.

`Expr3` and `Expr4` should inherit calibrated regimes rather than inventing new
ones independently.

### Expr3 V2 Guidance: Choose `p_meas` by Magnitude Matching

For `Expr3 V2`, the new axis is measurement bit-flip probability `p_meas`.

The design rule should follow the same logic used in `Expr2 V2` for choosing
correlated strength `g`:

- do not choose `p_meas` as an arbitrary inherited grid
- choose it so that the measurement-noise contribution is comparable to the
  already-confirmed `Expr2` composite channel strength
- avoid both extremes:
  - too small: measurement noise is only a negligible perturbation
  - too large: the experiment becomes mostly a measurement-noise benchmark

Current rough calibration note from confirmed `Expr2 V2` anchors:

- effective `2q` gate error probability is roughly in the `1e-3` range
- more specifically, the base `2q` gate term is around `5e-4` to `7e-4`
- with gate-specific mismatch included, the effective `p_2q` is typically
  around `1.0e-3` to `1.3e-3`

So the current working rule for `Expr3 V2` is:

- start `p_meas` near the same order of magnitude as effective `2q` gate error
- then extend upward by about one order of magnitude

Practical implication:

- the first `Expr3 V2` pilot should favor a grid centered around `1e-3`
- and extend up to roughly `1e-2`

Motivation:

- this range is more consistent with the already-calibrated `Expr2` gate scale
- it also better matches the intended NISQ-style interpretation than directly
  inheriting the old V1/legacy `p_meas = {0.01, 0.02, 0.05}` table without
  recalibration

Therefore, `Expr3 V2` parameter choice should be guided by:

- effective gate/composite budget from `Expr2`
- then matched `p_meas` magnitude
- not by legacy convenience values alone

### Expr3 V2 Recommended Starting Point

The first `Expr3 V2` pilot should inherit the top confirmed `Expr2` anchors
rather than rebuilding a fresh `(scale, f, g)` grid.

Recommended anchor set:

- `scale = 0.025`, `f = 1e4`, `g = 1.0`
- `scale = 0.025`, `f = 1e3`, `g = 1.6`
- `scale = 0.02`, `f = 1e2`, `g = 0.4`

Recommended first `p_meas` grid:

- `p_meas in {1e-3, 3e-3, 1e-2}`

Reasoning:

- `1e-3` is matched to the current effective `2q` gate-error scale
- `3e-3` is a moderate upward step without immediately turning measurement
  noise into the dominant channel
- `1e-2` is about one order of magnitude above the gate-matched baseline and
  should act as an upper robustness check

So the minimal `Expr3 V2 Phase A` pilot is:

- `3` confirmed `Expr2` anchors
- crossed with `3` measurement-noise values
- total `9` pilot conditions

Training rule:

- use the same stronger recipe already adopted in `Expr1 V2` and `Expr2 V2`
- do not fall back to legacy quick-scan budgets when introducing `p_meas`

### Expr3 V2 Objective

`Expr3 V2` should not be treated as "just add one more noise source and rerun."

Its role in the V2 story is:

- `Expr1` established that RL can beat `fixed_zero` in a gate-only setting
- `Expr2` established that this advantage can survive in a balanced
  gate-plus-correlated composite setting
- `Expr3` should test whether that advantage still survives after adding
  realistic measurement noise on top of the calibrated `Expr2` anchors

So the main question for `Expr3 V2` is:

- does the `Expr2` RL advantage remain visible in a fuller composite noise
  model that includes measurement bit-flips?

The intended claims are:

- primary claim:
  - some `Expr2`-positive regimes should remain RL-positive after adding
    measurement noise, if `p_meas` is kept within a physically matched range
- secondary claim:
  - increasing `p_meas` should eventually shrink or erase the RL-positive
    margin, which helps identify the robustness window of the learned
    gate-specific controller

Important interpretation note:

- `Expr3` is **not** meant to claim that RL directly controls or cancels the
  measurement-noise channel itself
- the control parameters are still gate-specific
- so the intended interpretation is robustness of gate-parameter optimization
  under fuller device noise, not direct suppression of measurement faults

In short, `Expr3 V2` is the bridge from:

- "RL works in calibrated gate-only and gate-plus-correlated settings"

to:

- "RL remains useful in a more realistic full composite setting, at least
  within a measurable measurement-noise window"

### Expr3 V2 Outcome

`Expr3 V2` is complete through:

- `Phase A pilot`
- `Phase B focused`
- `Phase C confirm`

Headline result:

- the `Expr2 V2` RL advantage does survive after adding measurement bit-flip
  noise, but the positive margin shrinks as `p_meas` grows

Confirmed ranking from `Phase C`:

1. `scale=0.025, f=1e4, g=1.0, p_meas=3e-3`
   - `improve(LER~) = +0.2703 +- 0.1003`
2. `scale=0.025, f=1e3, g=1.6, p_meas=1e-2`
   - `improve(LER~) = +0.1687 +- 0.0992`
3. `scale=0.025, f=1e4, g=1.0, p_meas=1e-2`
   - `improve(LER~) = +0.1123 +- 0.0967`

Interpretation:

- the best current `Expr3 V2` headline point is
  `scale=0.025, f=1e4, g=1.0, p_meas=3e-3`
- the best `Expr2` anchor remains the best choice after adding measurement
  noise
- when `p_meas` rises from `3e-3` to `1e-2`, the RL-positive margin weakens but
  does not disappear
- this supports the intended `Expr3` story:
  - gate-specific RL control remains useful in a fuller composite setting
  - but the robustness window narrows as measurement noise increases

Recommended GitHub-facing readout:

- best full-composite point:
  - `scale=0.025, f=1e4, g=1.0, p_meas=3e-3`
- shortest conclusion:
  - RL remains useful after adding measurement noise, but the advantage shrinks
    as readout faults become stronger

Figures and summaries:

- [Expr3 recap](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr3_full_composite_v2/expr3_v2_recap.md)
- [Phase A summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr3_full_composite_v2/phaseA_pilot/summary.json)
- [Phase B summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr3_full_composite_v2/phaseB_focused/summary.json)
- [Phase C summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr3_full_composite_v2/phaseC_confirm/summary.json)
- [Expr3 V2 Phase A figure](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr3_full_composite_v2/plots/expr3_v2_phaseA_p_sweep.png)
- [Expr3 V2 Phase B/C figure](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr3_full_composite_v2/plots/expr3_v2_phaseBC_compare.png)

### Expr3 Transfer Check

We also tested whether an `Expr1` gate-only learned control can be transplanted
directly into the confirmed `Expr3` full-composite conditions.

This transfer batch used `3` source seeds from the `Expr1 V2` gate-only
condition `ratio=1/10, scale=0.025`, and injected them into the top `3`
confirmed `Expr3` conditions.

Key result:

- the `Expr1-trained` transfer policy does **not** reliably explain the
  `Expr3` gains
- across all three confirm targets, `Expr3-trained` remains at least as good as
  transferred `Expr1-trained`
- in two of the three targets, the transfer policy is on average worse than
  `fixed_zero`

Current `3x3` transfer summary:

1. `scale=0.025, f=1e4, g=1.0, p_meas=3e-3`
   - transfer vs `fixed_zero`: `improve(LER~) = -0.0485 +- 0.1437`
   - transfer vs `Expr3-trained`: `delta success = -0.012695 +- 0.001406`
2. `scale=0.025, f=1e3, g=1.6, p_meas=1e-2`
   - transfer vs `fixed_zero`: `improve(LER~) = -0.0637 +- 0.1238`
   - transfer vs `Expr3-trained`: `delta success = -0.006402 +- 0.005945`
3. `scale=0.025, f=1e4, g=1.0, p_meas=1e-2`
   - transfer vs `fixed_zero`: `improve(LER~) = -0.1028 +- 0.0367`
   - transfer vs `Expr3-trained`: `delta success = +0.000000 +- 0.001381`

Interpretation:

- `Expr3` gains should not be described as simple reuse of an `Expr1`
  gate-only policy
- by the time measurement noise is added, gate-only transfer is no longer a
  reliable explanation for the observed full-composite advantage
- the safer conclusion is that `Expr3-trained` policies require additional
  adaptation to the fuller composite environment

Supporting files:

- [Expr1->Expr3 transfer summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/policy_transfer_expr3/transfer_summary.md)
- [Expr1->Expr3 transfer comparison](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/policy_transfer_expr3/transfer_compare.png)
- [Expr1->Expr3 transfer deltas](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/policy_transfer_expr3/transfer_deltas.png)

### Expr4 V2 Design: Cycle-Decay Under Fixed Full-Composite Test Noise

`Expr4 V2` should be treated as a fixed-policy cycle sweep, but the policy
comparison should be updated relative to V1.

Recommended V2 comparison:

- `full_channel_RL`
- `full_channel_transfer_expr1`
- `full_channel_fixed_zero`

All three policies should be evaluated under the same fixed full-composite test
channel while sweeping only the QEC-cycle count.

This means:

- choose one fixed `Expr3 V2` full-composite showcase condition
- hold the test noise fixed
- hold all policies fixed
- vary only `n_rounds`

Recommended showcase condition for the first `Expr4 V2` run:

- inherit the best confirmed `Expr3 V2` point:
  - `scale = 0.025`
  - `f = 1e4`
  - `g = 1.0`
  - `p_meas = 3e-3`

So the intended policy definitions are:

- `full_channel_RL`
  - source policy trained directly on the chosen `Expr3 V2` full-composite
    showcase condition
- `full_channel_transfer_expr1`
  - source policy taken from the best `Expr1 V2` gate-only checkpoint and
    transferred without retraining into the same full-composite test condition
- `full_channel_fixed_zero`
  - zero-action baseline evaluated under the same full-composite condition

Why this is preferable to the V1 comparison:

- V1 mainly asked whether full-composite-aware training beats
  composite-only training at long cycle depth
- V2 already established a more important distinction:
  - gate-only transfer does not explain the gains seen in `Expr2/Expr3`
- so `Expr4 V2` should extend that exact story into the long-memory regime

Main question for `Expr4 V2`:

- as QEC cycle count increases, does the full-composite-trained controller
  preserve logical performance better than a transferred gate-only policy and
  better than `fixed_zero`?

Recommended cycle sweep:

- `n_rounds in {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}`

Recommended plotted quantities:

- main figure:
  - x-axis = `n_rounds`
  - y-axis = `success_rate` or `logical_observable_proxy = 2 * success_rate - 1`
  - lines:
    - `full_channel_RL`
    - `full_channel_transfer_expr1`
    - `full_channel_fixed_zero`
- companion figure:
  - x-axis = `n_rounds`
  - y-axis = `LER~`
- analysis figure:
  - `LER~(fixed_zero) - LER~(full_channel_RL)`
  - `LER~(transfer_expr1) - LER~(full_channel_RL)`

### Expr4 V2 Training Rule

For `Expr4 V2`, the training budget used to obtain `full_channel_RL` should be
at least as strong as the `Expr3 V2` confirm budget, and can reasonably be
made slightly stronger because this policy will act as the showcase policy for
the full cycle sweep.

Minimum acceptable source-policy recipe:

- `total_timesteps = 4096`
- `rollout_steps = 64`
- `steane_shots_per_step = 16`
- `ppo_learning_rate = 1e-4`
- `ppo_ent_coef = 1e-3`
- `steane_action_penalty_coef = 0.005`
- `steane_miscal_penalty_coef = 0.001`
- `trace_finetune_timesteps = 512`

Preferred stronger showcase recipe for `full_channel_RL`:

- `total_timesteps = 8192`
- `rollout_steps = 64` or `128`
- `steane_shots_per_step = 16` or `24`
- `ppo_learning_rate = 1e-4`
- `ppo_ent_coef = 1e-3`
- `steane_action_penalty_coef = 0.005`
- `steane_miscal_penalty_coef = 0.001`
- `trace_finetune_timesteps = 1024`

Practical interpretation:

- `Expr4` should not spend most of its budget on many training conditions
- instead, it should spend budget on one very strong full-channel source policy
  plus high-quality fixed-policy evaluation over the cycle sweep

### Expr4 V2 Outcome

`Expr4 V2` is now complete through `Phase C confirm`.

Final headline condition:

- `scale = 0.025`
- `f = 1e4`
- `g = 1.0`
- `p_meas = 3e-3`

Final confirm used `6` seeds and compared:

- `full_channel_RL`
- `full_channel_transfer_expr1`
- `fixed_zero`

Confirmed cycle-sweep readout:

- `n_rounds = 5`
  - `full_channel_RL`: `97.82% +- 0.31%`
  - `full_channel_transfer_expr1`: `97.00% +- 0.69%`
  - `fixed_zero`: `97.06% +- 0.56%`
- `n_rounds = 25`
  - `full_channel_RL`: `91.36% +- 0.72%`
  - `full_channel_transfer_expr1`: `86.50% +- 1.64%`
  - `fixed_zero`: `86.20% +- 2.03%`
- `n_rounds = 50`
  - `full_channel_RL`: `82.67% +- 1.69%`
  - `full_channel_transfer_expr1`: `74.64% +- 3.40%`
  - `fixed_zero`: `74.99% +- 3.00%`

Interpretation:

- the full-channel-trained controller remains best across the whole cycle sweep
- the gap relative to `Expr1` transfer becomes substantial at longer cycle
  depth
- `Expr1` transfer is not a reliable explanation for the long-cycle advantage
  seen under the full composite test channel

Recommended GitHub-facing readout:

- best long-cycle headline condition:
  - `scale=0.025, f=1e4, g=1.0, p_meas=3e-3`
- shortest conclusion:
  - full-channel RL keeps a clear advantage over both `Expr1` transfer and
    `fixed_zero`, and that gap widens at longer cycle depth

Figures and summaries:

- [Expr4 Phase A summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/summary.md)
- [Expr4 Phase A figure](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/expr4_v2_cycle_decay.png)
- [Expr4 Phase B summary `p=1e-2, f=1e4, g=1.0`](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseB_focus_p010_f1e4_g10/summary.md)
- [Expr4 Phase B summary `p=1e-2, f=1e3, g=1.6`](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseB_focus_p010_f1e3_g16/summary.md)
- [Expr4 Phase C summary](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseC_confirm/summary.md)
- [Expr4 Phase C figure](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseC_confirm/cycle_decay.png)

## Suggested Directory Layout

Initial V2 layout:

- `README.md`
- `notes/`
- `expr1_gate_only_v2/`
- `expr2_standard_composite_v2/`
- `expr3_full_composite_v2/`
- `expr4_cycle_decay_full_composite_v2/`

These subfolders do not all need to be populated immediately.

## Current Recommendation

Yes, running one experiment at a time is the right move.

The current recommendation is:

1. freeze V1 as historical record
2. do not reinterpret V1 as the final benchmark truth
3. start V2 from `Expr1` recalibration
4. promote later experiments only after the simpler one is physically well-calibrated

That will give the project a cleaner benchmark story and reduce the chance of
repeating the same regime-selection mistake across multiple experiment families.



------------------------------------------------------------------------------------
------------------------------------------------------------------------------------


## Takeaway Observations (🔥🔥🔥🔥🔥🔥)

This section logs the main observations after the first `Expr1 V2` numerical
reruns were completed.

### 1. Stronger Training Clearly Helps in Focused `Expr1 V2`

Using the stronger training recipe

- `total_timesteps = 4096`
- `steane_shots_per_step = 16`
- `rollout_steps = 64`
- `ppo_learning_rate = 1e-4`
- `ppo_ent_coef = 0.001`
- `steane_action_penalty_coef = 0.005`
- `steane_miscal_penalty_coef = 0.001`

improved the focused `Phase B` rerun on all selected conditions relative to the
older `Phase B` baseline.

Observed aggregate changes:

- `1q/2q = 1/1`, `scale = 0.5`:
  - `improve(LER~): 0.1639 -> 0.2553`
  - `learned.success_rate: 0.9604 -> 0.9644`
- `1q/2q = 1/10`, `scale = 0.1`:
  - `improve(LER~): 0.2951 -> 0.3828`
  - `learned.success_rate: 0.9140 -> 0.9275`
- `1q/2q = 1/10`, `scale = 0.25`:
  - `improve(LER~): 0.2352 -> 0.2788`
  - `learned.success_rate: 0.7752 -> 0.7850`
- `1q/2q = 1/10`, `scale = 0.5`:
  - `improve(LER~): 0.0177 -> 0.1100`
  - `learned.success_rate: 0.5724 -> 0.6070`

Interpretation:

- part of the earlier `learned < fixed_zero` issue was indeed a training-budget /
  signal-quality issue
- once training is made less noisy and less brittle, PPO more reliably beats
  the zero-action baseline on the focused conditions

### 2. Full `Phase A` Improved, but the Improvement Is Not Uniform

The full stronger `Phase A` candidate rerun also improved several important
conditions, but the effect is mixed across the whole grid.

Important positive changes:

- `1q/2q = 1/10`, `scale = 0.025`:
  - `improve(LER~): -1.8500 -> +0.4667`
- `1q/2q = 1/10`, `scale = 0.1`:
  - `improve(LER~): -0.0219 -> +0.2009`
- `1q/2q = 1/1`, `scale = 0.1`:
  - `improve(LER~): -0.6000 -> +0.2667`
- `1q/2q = 1/1`, `scale = 0.25`:
  - `improve(LER~): +0.3000 -> +0.7000`
- `1q/2q = 1/1`, `scale = 0.5`:
  - `improve(LER~): +0.0733 -> +0.4389`

Important remaining limitation:

- the hardest candidate condition, `1q/2q = 1/10`, `scale = 0.5`, still did
  not become a robust RL-positive point
  - `improve(LER~): -0.0299 -> -0.0117`
  - `learned.success_rate: 0.6521 -> 0.6802`

Interpretation:

- stronger training improves the controller even in difficult regimes
- but for the heaviest `1/10` endpoint, the gain is still not strong enough to
  reliably overcome `fixed_zero`
- this suggests that the problem is not only "training was too weak"; the
  regime itself is likely close to the edge of what should be treated as a
  useful calibration point for `Expr1`

### 3. Very Light Regimes Are Still Poor Benchmark Points

Several low-scale conditions remain close to perfect success:

- `1q/2q = 1/1`, `scale = 0.001`
- `1q/2q = 1/1`, `scale = 0.0025`
- `1q/2q = 1/10`, `scale = 0.001`

In these conditions:

- `fixed_zero.success_rate` is already near `1`
- `improve(LER~)` becomes numerically unstable and seed-sensitive
- positive or negative relative changes are not very informative

Interpretation:

- these points are too light for strong RL claims
- they can remain as calibration references, but they should not carry much
  weight in the final benchmark narrative

### 4. Current `Expr1 V2` Readout

The current evidence supports the following `Expr1 V2` story:

- `learned < fixed_zero` was not purely a physics/regime-selection issue
- improving training budget and signal quality materially improves RL outcomes
- the middle part of the grid now looks much more like a learnable RL regime
- the lightest points are too easy
- the heaviest `1q/2q = 1/10`, `scale = 0.5` point is still too hard or too
  unstable to serve as a clean RL-positive benchmark

So the working lesson from `Expr1 V2` is:

- keep the mid-scale points
- treat the very light points as calibration only
- treat the heaviest `1/10` endpoint as a boundary/failure case, not as a core
  evidence point

### 5. Practical Next-Step Recommendation

For `Expr1` itself:

- use the stronger-training recipe as the new default for any serious rerun
- keep `Phase B focused` as the main evidence figure for "RL can beat
  `fixed_zero` under gate-only noise"
- use the full `Phase A` grid mainly to mark:
  - too light
  - usable
  - too heavy / unstable

For downstream V2 work:

- carry the stronger-training lesson into `Expr2`
- do not assume that every negative point in old quick scans was a meaningful
  counterexample to RL
- but also do not overclaim that stronger training rescues every regime;
  the heaviest `Expr1` endpoint remains a warning sign

### 6. `Expr2` Gains Are Not Explained by Simple `Expr1` Policy Transfer

We explicitly tested whether an `Expr1` gate-only learned controller can be
transferred into the top confirmed `Expr2` composite conditions.

This transfer batch used:

- `3` `Expr1 V2` source seeds from `1q/2q = 1/10`, `scale = 0.025`
- `3` confirmed `Expr2` target conditions

Observed transfer summary:

- `scale = 0.025`, `f = 1e4`, `g = 1.0`
  - transfer vs `fixed_zero`: `improve(LER~) = -0.0758 +- 0.1553`
  - transfer vs `Expr2-trained`: `delta success = -0.003581 +- 0.000797`
- `scale = 0.025`, `f = 1e3`, `g = 1.6`
  - transfer vs `fixed_zero`: `improve(LER~) = +0.0194 +- 0.2538`
  - transfer vs `Expr2-trained`: `delta success = -0.011176 +- 0.007997`
- `scale = 0.02`, `f = 1e2`, `g = 0.4`
  - transfer vs `fixed_zero`: `improve(LER~) = -0.0955 +- 0.2309`
  - transfer vs `Expr2-trained`: `delta success = -0.006293 +- 0.003609`

Interpretation:

- `Expr1-trained` transfer is not a reliable replacement for `Expr2-trained`
  control
- across all three confirmed `Expr2` targets, `Expr2-trained` remains better
  than transferred `Expr1-trained`
- in two of the three targets, transferred `Expr1-trained` is on average even
  worse than `fixed_zero`

So the working lesson from this transfer check is:

- `Expr2` gains should not be described as simple reuse of a gate-only policy
- the safer claim is that `Expr2-trained` policies learn additional adaptation
  to the composite noise environment
- this supports a composite-specific RL story, but it does not yet isolate the
  exact mechanism as direct suppression of the correlated Pauli component

### 7. `Expr4 V2` Shows That Long-Cycle Advantage Also Requires Full-Channel Training

We extended the same transfer question into the cycle-decay setting by fixing
one full-composite test channel and sweeping only `n_rounds`.

Final `Phase C` confirm on
`scale = 0.025, f = 1e4, g = 1.0, p_meas = 3e-3` showed:

- `n_rounds = 25`
  - `full_channel_RL`: `91.36% +- 0.72%`
  - `full_channel_transfer_expr1`: `86.50% +- 1.64%`
  - `fixed_zero`: `86.20% +- 2.03%`
- `n_rounds = 50`
  - `full_channel_RL`: `82.67% +- 1.69%`
  - `full_channel_transfer_expr1`: `74.64% +- 3.40%`
  - `fixed_zero`: `74.99% +- 3.00%`

Interpretation:

- full-channel RL remains clearly best over the entire cycle sweep
- the advantage is small at short depth but becomes much larger at longer
  cycle count
- transferred `Expr1` gate-only control does not explain the long-cycle
  robustness of the full-channel-trained policy

So the working lesson from `Expr4 V2` is:

- the full-composite advantage is not limited to fixed short-depth evaluation
- it persists into the long-cycle regime
- and long-cycle robustness should be treated as additional evidence that
  `Expr1` gate-only transfer is insufficient once the full channel is present
