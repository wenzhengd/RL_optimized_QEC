# RL Steane Tune Experiment Summary

## Completion Status

- Expr1: Phase A complete; Phase B/C not yet run.
- Expr2: Phase A/B/C complete.
- Expr3: Phase A/B/C complete.
- Expr4: Phase A/B/C complete for cycle-sweep showcase conditions.

## Best Staged Conditions

- expr2 phaseC_confirm: `expr2c_confirm_f1e3_g16` improve(LER~)=+30.96% +- 14.36%, learned_success=91.53%.
- expr3 phaseC_confirm: `expr3c_confirm_p001_f1e2_g16` improve(LER~)=+21.63% +- 6.66%, learned_success=87.81%.

## Expr4 Final Showcase

- n_rounds=5: full=0.7921 +- 0.0248, composite=0.8688 +- 0.0214
- n_rounds=25: full=0.3105 +- 0.0568, composite=0.3949 +- 0.0493
- n_rounds=50: full=0.0785 +- 0.0478, composite=0.1166 +- 0.0417

## Readout

- Under the final Expr4 showcase condition `p=0.01, f=1e2, g=1.6`, `trained_on_composite` remains above `trained_on_full_composite` across the cycle sweep.
