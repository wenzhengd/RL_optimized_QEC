# RL Steane Tuning Experiments

This folder defines a concrete 4-part experiment plan for demonstrating that RL
gate-control optimization improves Steane-QEC performance under progressively
harder noise models.

The plan is intentionally aligned with the current project terminology:

- `google_gate_specific` = gate-noise-only baseline
- `composed_google_gate_specific_correlated` = standard composite
- `full composite` = standard composite + measurement bit-flip overlay

See also:

- [code/rl_train/README.md](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/rl_train/README.md)
- [code/quantum_simulation/noise_modeling.md](/Users/wenzhengdong/Desktop/RL_QEC_control_tuning/code/quantum_simulation/noise_modeling.md)

## Common Rules

### Metrics

For every experiment, do not report only final success rate.

Primary metrics:

- `success_rate`
- `LER~ = 1 - success_rate`
- `improvement_vs_fixed_zero.ler_proxy`

Secondary metrics:

- `detector_rate` or `DR` proxy
- seed-level positive-gain ratio
- mean +- std across seeds

### Evaluation Protocol

Unless explicitly changed, use one fixed evaluation protocol for all compared policies:

- same `post_eval_episodes`
- same `eval_steane_shots_per_step`
- same test-noise distribution
- same seed list

This is critical. Different training distributions may be compared, but test
distribution must be fixed inside each comparison figure.

Recommended default evaluation settings:

- quick / focused runs:
  - `post_eval_episodes = 8`
  - `eval_steane_shots_per_step = 24` or `32`
- confirm runs:
  - `post_eval_episodes = 32`
  - `eval_steane_shots_per_step = 64`

### Shots Per Candidate

In the current codebase, "shots per candidate" means:

- training-time shots per RL action candidate:
  - `steane_shots_per_step`
- evaluation-time shots per candidate:
  - `eval_steane_shots_per_step`

In `candidate_eval` mode, one RL step does:

- propose one control action candidate
- hold that candidate fixed for the full simulated memory experiment
- evaluate it with `shots_per_step` Monte Carlo shots
- average the resulting statistics into the reward / summary metrics

Recommended practical values for this experiment family:

- quick scan:
  - `steane_shots_per_step = 4`
  - `eval_steane_shots_per_step = 24`
- focused comparison:
  - `steane_shots_per_step = 4` or `6`
  - `eval_steane_shots_per_step = 24` or `32`
- confirm runs:
  - keep training shots moderate
  - increase evaluation shots to `32` or `64`

Reason:

- training should stay cheap enough to sweep many conditions
- evaluation should be more stable than training
- project practice already supports this split well

### Correlated-Channel Convention

For all correlated/composite experiments, fix:

- `steane_channel_corr_g_mode = per_circuit`

Recommended `(f, g)` grid from current project practice:

- `f in {1e2, 1e3, 1e4}`
- `g in {0.4, 1.0, 1.6}`

These correspond to:

- low / mid / high correlation frequency
- weak / mid / strong correlated-channel strength

### Policy Labels

Use precise policy labels in plots/tables:

- `trained_on_gate`
- `trained_on_composite`
- `trained_on_full_composite`
- `fixed_zero`

Avoid ambiguous labels such as:

- `trained`
- `partially-trained`
- `untrained`

`fixed_zero` is the preferred baseline name for "no control optimization".

## Experiment 1: Gate-Noise-Only RL Works

### Goal

Show that RL control optimization improves QEC performance when the only active
noise family is Google-like gate noise.

### Train Noise Distribution

- `steane_noise_channel = google_gate_specific`
- `steane_measurement_bitflip_prob = 0.0`

### Test Noise Distribution

Same as train:

- `google_gate_specific`
- no correlated noise
- no measurement overlay

### Sweep Definition

Do not use an abstract "global error strength" without mapping it to code
parameters.

Use explicit regime parameters:

- hold a ratio between 1q and 2q scaling
- sweep overall scale through `channel_regime_a` / `channel_regime_b`

Recommended two ratio slices:

1. `1q / 2q = 1 / 4`
   - example sweep:
     - `channel_regime_a in {0.25, 0.5, 1.0, 1.5, 2.0}`
     - `channel_regime_b = 4 * channel_regime_a`
2. `1q / 2q = 4 / 1`
   - example sweep:
     - `channel_regime_b in {0.25, 0.5, 1.0, 1.5, 2.0}`
     - `channel_regime_a = 4 * channel_regime_b`

If that is too wide, reduce to 3 scales first and expand later.

### Comparison

Compare:

- `trained_on_gate`
- `fixed_zero`

This experiment is an in-distribution baseline/sanity experiment, not a
transfer experiment.

### Recommended Figures

Main figure:

- line plot: x = overall gate-noise scale, y = `success_rate`
- two panels:
  - `1q/2q = 1/4`
  - `1q/2q = 4/1`

Recommended companion figure:

- same x-axis, y = `improvement_vs_fixed_zero.ler_proxy`

## Experiment 2: RL Under Standard Composite Noise

### Goal

Show that a policy trained under standard composite noise performs better on
composite-noise test conditions than:

- a gate-only-trained policy
- the fixed-zero baseline

### Train Noise Distributions

Policy A:

- `trained_on_composite`
- `steane_noise_channel = composed_google_gate_specific_correlated`
- `steane_measurement_bitflip_prob = 0.0`

Policy B:

- `trained_on_gate`
- `steane_noise_channel = google_gate_specific`
- `steane_measurement_bitflip_prob = 0.0`

Baseline:

- `fixed_zero`

### Test Noise Distribution

Fixed composite test distribution:

- `steane_noise_channel = composed_google_gate_specific_correlated`
- `steane_measurement_bitflip_prob = 0.0`
- `(f, g)` grid fixed to:
  - `f in {1e2, 1e3, 1e4}`
  - `g in {0.4, 1.0, 1.6}`
- `steane_channel_corr_g_mode = per_circuit`

### Comparison

Under the same composite test condition, compare:

- `trained_on_composite`
- `trained_on_gate`
- `fixed_zero`

This is a transfer/generalization comparison. The key question is whether
composite-aware training is necessary once correlated noise is present.

### Recommended Figures

Main figure:

- 3 heatmaps over `(g, f)`:
  - `trained_on_composite.success_rate`
  - `trained_on_gate.success_rate`
  - `fixed_zero.success_rate`

Better summary figure:

- heatmap of `improvement_vs_fixed_zero.ler_proxy`
  for `trained_on_composite`

Recommended comparison figure:

- heatmap of
  `LER~(trained_on_gate) - LER~(trained_on_composite)`
  over `(g, f)`

This directly shows where composite-aware RL training helps beyond gate-only RL.

## Experiment 3: RL Under Full Composite Noise

### Goal

Show that when measurement error is added, a policy trained under full
composite noise performs better on full-composite test conditions than:

- a policy trained only on standard composite noise
- the fixed-zero baseline

### Train Noise Distributions

Policy A:

- `trained_on_full_composite`
- `steane_noise_channel = composed_google_gate_specific_correlated`
- `steane_measurement_bitflip_prob = p_meas`

For this experiment, train one separate `trained_on_full_composite` policy for
each fixed `p_meas` in the selected sweep.

Policy B:

- `trained_on_composite`
- `steane_noise_channel = composed_google_gate_specific_correlated`
- `steane_measurement_bitflip_prob = 0.0`

Baseline:

- `fixed_zero`

### Test Noise Distribution

Fixed full-composite distribution:

- `steane_noise_channel = composed_google_gate_specific_correlated`
- correlated grid same as Experiment 2:
  - `f in {1e2, 1e3, 1e4}`
  - `g in {0.4, 1.0, 1.6}`
- measurement overlay:
  - `p_meas in {0.01, 0.02, 0.05}`
- `steane_channel_corr_g_mode = per_circuit`

### Comparison

For each fixed `p_meas`, compare on the same full-composite test distribution:

- `trained_on_full_composite`
- `trained_on_composite`
- `fixed_zero`

This is again a train-distribution comparison, now testing whether
measurement-aware training matters once readout error is present.

### Recommended Figures

Main figure:

- one heatmap panel per `p_meas`
- color = `improvement_vs_fixed_zero.ler_proxy`
- shown for `trained_on_full_composite`

Recommended comparison figure:

- one heatmap panel per `p_meas`
- color =
  `LER~(trained_on_composite) - LER~(trained_on_full_composite)`
- this directly shows the extra value of measurement-aware RL training

Recommended summary figure:

- x-axis = `p_meas`
- y-axis = average `improvement_vs_fixed_zero.ler_proxy`
- separate lines for:
  - `trained_on_full_composite`
  - `trained_on_composite`

## Experiment 4: Performance Decay vs QEC Cycles Under Full Composite Noise

### Goal

Show how logical-memory performance degrades as QEC depth increases under one
fixed full-composite noise condition, and whether the RL-trained controller
slows that degradation relative to simpler baselines.

This experiment is intended to play the role of a "memory-decay curve" figure:

- x-axis = QEC cycles
- y-axis = logical performance metric

### Core Design Choice

This experiment should be a **fixed-policy evaluation sweep**, not a retraining
sweep.

That is:

- train the policies once at one chosen training distribution
- then hold the learned policies fixed
- vary only the evaluation-time `n_rounds`

Do **not** retrain a new policy for every QEC-cycle value. That would answer a
different question.

### Train Noise Distributions

Policy A:

- `trained_on_full_composite`
- `steane_noise_channel = composed_google_gate_specific_correlated`
- `steane_measurement_bitflip_prob = p_meas_fixed`

Policy B:

- `trained_on_composite`
- `steane_noise_channel = composed_google_gate_specific_correlated`
- `steane_measurement_bitflip_prob = 0.0`

Baseline:

- `fixed_zero`

Recommended default:

- use this fixed primary showcase condition:
  - `f = 1e2`
  - `g = 0.1`
  - `p_meas_fixed = 0.01`

If runtime allows, repeat the same cycle-sweep figure for one additional harsher
condition, e.g. `p_meas_fixed = 0.05`.

### Test Noise Distribution

Fix one full-composite test condition:

- `steane_noise_channel = composed_google_gate_specific_correlated`
- `steane_channel_corr_g_mode = per_circuit`
- fixed correlated parameters:
  - `f = f_fixed`
  - `g = g_fixed`
- fixed measurement overlay:
  - `p_meas = p_meas_fixed`

Then sweep only:

- `n_rounds in {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}`

### Comparison

Under the same fixed full-composite test condition and for every QEC-cycle
value, compare:

- `trained_on_full_composite`
- `trained_on_composite`
- `fixed_zero`

Main question:

- does the RL-trained policy preserve logical performance better as memory depth
  increases?

Secondary question:

- does full-composite-aware training outperform standard-composite-only training
  at long QEC depth?

### Metrics

Primary plotting metric:

- `success_rate`

Recommended transformed metric for a more "logical observable"-style plot:

- `logical_observable_proxy = 2 * success_rate - 1`

Also report:

- `LER~ = 1 - success_rate`
- `improvement_vs_fixed_zero.ler_proxy`

### Recommended Figures

Main figure:

- line plot:
  - x = `n_rounds` (QEC cycles)
  - y = `logical_observable_proxy`
- lines:
  - `trained_on_full_composite`
  - `trained_on_composite`
  - `fixed_zero`

Recommended companion figure:

- line plot:
  - x = `n_rounds`
  - y = `LER~`

Recommended analysis view:

- line plot:
  - x = `n_rounds`
  - y = `LER~(baseline) - LER~(policy)`
- at minimum:
  - `fixed_zero - trained_on_full_composite`
  - `trained_on_composite - trained_on_full_composite`

### Evaluation Notes

- For this experiment, use `n_rounds` as the QEC-cycle axis.
- Do not use raw simulator `n_steps` as the plotted x-axis.
- In Steane, `n_steps = 6 * n_rounds`, so `n_rounds` is the physically cleaner
  quantity to show.
- This experiment is best run with a stronger evaluation protocol than the quick
  scans:
  - `post_eval_episodes = 32`
  - `eval_steane_shots_per_step = 64`
  - and a fixed seed list across all cycle values

## Minimum Statistical Protocol

To keep the plan executable, use a staged protocol instead of jumping directly
to high-cost runs.

### Phase A: quick scan

- seeds: `5`
- use small budget to eliminate clearly bad regions

### Phase B: focused comparison

- seeds: `10`
- use the shortlisted conditions only

### Phase C: confirm

- seeds: `20`
- only for the top conditions / main paper figures

For every reported main figure, note:

- number of seeds
- `post_eval_episodes`
- `eval_steane_shots_per_step`
- whether metrics come from fast eval or trace eval

## Recommended Output Layout

Keep one folder per experiment stage, for example:

- `expr1_gate_only/`
- `expr2_standard_composite/`
- `expr3_full_composite/`
- `expr4_cycle_decay_full_composite/`

Inside each:

- `README.md`
- `summary.json`
- optional `plots/`
- optional per-condition subfolders

## Main Risks

1. If train and test distributions are not separated clearly, conclusions become ambiguous.
2. If only `success_rate` is reported, the work will not align with the rest of the project.
3. If `(f, g)` or `p_meas` are described qualitatively but not fixed numerically, results will drift.
4. If each figure uses a different eval protocol, comparisons become unreliable.

## Practical Next Step

The most practical order is:

1. finalize Experiment 1 parameter sweep
2. freeze the Experiment 2 `(f, g)` grid
3. re-use the same `(f, g)` grid in Experiment 3
4. select the top full-composite condition for Experiment 4
5. only then decide the exact budget for the `5/10/20`-seed phases

## Recommended Execution Order

Use the four experiments in the following order:

1. `Experiment 1`
   - establish the gate-noise-only sanity baseline
2. `Experiment 2`
   - establish the standard-composite comparison
3. `Experiment 3`
   - establish the full-composite comparison
4. `Experiment 4`
   - only after a fixed full-composite showcase condition has been selected

Inside each experiment, use the three-phase protocol:

1. `Phase A: quick scan`
   - low-cost elimination of weak conditions
2. `Phase B: focused comparison`
   - compare only shortlisted conditions
3. `Phase C: confirm`
   - run final reporting conditions with strongest statistics

Recommended practical rollout:

1. run `Expr1-A`
2. run `Expr2-A`
3. run `Expr3-A`
4. shortlist the best conditions
5. run `Expr2-B/C`
6. run `Expr3-B/C`
7. run `Expr4-A/B/C` on the final chosen showcase condition

Reason:

- `Experiment 1` is mainly a sanity baseline
- the main comparative scientific value sits in `Experiment 2` and `Experiment 3`
- `Experiment 4` is a presentation/interpretation experiment and should be run
  only after the showcase full-composite condition is fixed
