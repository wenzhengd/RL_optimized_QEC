# Final Experiment Conclusion

## Scope

This report summarizes the completed RL Steane tuning experiments under
`code/data_generated/rl_Steane_tune_experiment/`, using the generated tables in:

- `staged_summary.csv`
- `expr4_cycle_summary.csv`
- `experiment_summary.md`

and the figure outputs in:

- `../plots/expr1_phaseA_lines.png`
- `../plots/expr2_phaseA_heatmap.png`
- `../plots/expr3_phaseA_heatmaps.png`
- `../plots/expr4_phaseC_cycle_decay.png`

## Completion Status

Completed:

- `Expr2` Phase `A/B/C`
- `Expr3` Phase `A/B/C`
- `Expr4` Phase `A/B/C`

Partially completed:

- `Expr1` Phase `A` only

Interpretation:

- The main composite-noise and full-composite-noise experimental line is complete.
- The gate-only baseline line is sufficient for sanity checking, but not fully closed out to the same depth.

## Main Findings

### 1. Gate-only RL shows a real but regime-dependent positive signal

From `Expr1 Phase A`, RL can improve performance under gate-only noise, but the gain depends strongly on regime.

Best quick-scan conditions:

- `expr1a_ratio_4to1_scale_05`: improve(LER~) `+14.83%`
- `expr1a_ratio_1to4_scale_05`: improve(LER~) `+10.45%`

Weaker or negative conditions also appear at heavier scales, so the gate-only story is:

- RL helps in some gate-noise regimes
- the effect is not uniformly strong across the full sweep

This is enough to support the role of `Expr1` as a sanity baseline.

### 2. Standard composite training gives a strong and reproducible gain

`Expr2` is the cleanest positive result in the whole experiment set.

Best confirm condition:

- `expr2c_confirm_f1e3_g16`
- improve(LER~): `+30.96% +- 14.36%`
- learned success: `91.53% +- 1.45%`

Focused-stage results were also consistent:

- `f=1e3, g=1.6`: `+37.22%`
- `f=1e3, g=0.4`: `+34.15%`
- `f=1e2, g=1.0`: `+32.27%`

Conclusion:

- RL control optimization is clearly useful once correlated composite noise is present.
- The effect survives through quick, focused, and confirm stages.

### 3. Full-composite training improves over fixed-zero, but not beyond composite-only training

`Expr3` confirms that full-composite-trained policies still beat the fixed-zero baseline.

Best confirm condition:

- `expr3c_confirm_p001_f1e2_g16`
- improve(LER~): `+21.63% +- 6.66%`
- learned success: `87.81% +- 1.08%`

So the positive result is:

- full-composite-aware training is beneficial relative to no control optimization

But the stronger claim is *not* supported:

- we do not have evidence here that `trained_on_full_composite` is better than `trained_on_composite`

At least in the current experiment set, the measurement-aware training benefit did not dominate the simpler composite-only policy.

### 4. The cycle-decay experiment is a stable negative result for the full-composite advantage hypothesis

`Expr4` was first tried on the README default showcase condition:

- `p=0.01, f=1e2, g=0.1`

That condition did **not** show `trained_on_full_composite` beating
`trained_on_composite`.

We then switched to the stronger full-composite showcase condition selected from
`Expr3`:

- `p=0.01, f=1e2, g=1.6`

This was run through:

- Phase B: `10` seeds
- Phase C: `20` seeds

The final `20-seed` cycle-decay result still shows `trained_on_composite` above
`trained_on_full_composite` across the full sweep:

- `n_rounds=5`
  - full: `0.7921`
  - composite: `0.8688`
- `n_rounds=25`
  - full: `0.3105`
  - composite: `0.3949`
- `n_rounds=50`
  - full: `0.0785`
  - composite: `0.1166`

This is the most important negative result of the study:

- under the final showcase condition, memory-decay performance does **not**
  support the hypothesis that full-composite training outperforms composite-only
  training

## Overall Interpretation

The experimental evidence supports the following statement:

1. RL control tuning is effective for Steane-QEC under composite correlated noise.
2. That gain is robust relative to the fixed-zero controller.
3. Adding measurement-error-aware training does not automatically yield a better
   policy than standard composite training.
4. In the final memory-decay experiment, the composite-only trained policy
   remains stronger than the full-composite-trained policy, even after a
   `20-seed` confirm run.

So the strongest defensible headline is:

> RL improves Steane-QEC control under composite noise, but the current data do
> not show that measurement-aware training beats composite-only training.

## Recommended Writeup Position

For any paper / memo / presentation draft, the safest framing is:

- Present `Expr2` as the main positive result.
- Present `Expr3` as a qualified result: improvement over fixed baseline, but no
  confirmed superiority over composite-only training.
- Present `Expr4` as a memory-decay stress test that failed to validate the
  full-composite advantage hypothesis.

Avoid claiming:

- that full-composite training is the best policy family overall
- that measurement-aware RL preserves logical memory better than
  composite-only RL

## Remaining Gaps

The main unfinished items are:

1. `Expr1` Phase `B/C` were not completed.
2. The visualization set is sufficient for internal reporting, but not yet
   polished into publication-ready multi-panel figures.
3. No formal statistical hypothesis test was added beyond seed-wise mean/std
   reporting.

These are follow-up tasks, not blockers for stating the core conclusion above.
