# Expr4 V2 Plan

## Objective

Evaluate memory-performance decay under one fixed full-composite test channel
while comparing three fixed strategies:

- `full_channel_RL`
- `full_channel_transfer_expr1`
- `full_channel_fixed_zero`

All three are evaluated on the same full-composite test condition. Only
evaluation-time `n_rounds` is swept.

## Recommended Showcase Condition

Use the best confirmed `Expr3 V2` condition:

- `scale = 0.025`
- `f = 1e4`
- `g = 1.0`
- `p_meas = 3e-3`

## Policies

- `full_channel_RL`
  - trained directly on the showcase full-composite condition
- `full_channel_transfer_expr1`
  - load the best `Expr1 V2` gate-only checkpoint and evaluate it without
    retraining on the same showcase full-composite condition
- `full_channel_fixed_zero`
  - zero-action baseline under the same showcase full-composite condition

## Cycle Sweep

- `n_rounds in {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}`

## Recommended Source Checkpoint

For the first V2 run, use:

- `Expr1` source checkpoint:
  - `code/data_generated/rl_steane_tune_experiments_V2/policy_transfer/expr1_source_seed720_checkpoint.pt`

This keeps the first cycle-sweep run simple and reproducible. Multi-seed source
extensions can be added later.

## Recommended Command

```bash
PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  -m rl_train.benchmarks.eval_steane_cycle_sweep_v2 \
  --code-family steane \
  --device cpu \
  --steane-noise-channel composed_google_gate_specific_correlated \
  --steane-channel-regime-a 0.025 \
  --steane-channel-regime-b 0.25 \
  --steane-channel-corr-f 10000.0 \
  --steane-channel-corr-g 1.0 \
  --steane-channel-corr-g-mode per_circuit \
  --steane-measurement-bitflip-prob 0.003 \
  --total-timesteps 8192 \
  --rollout-steps 64 \
  --ppo-learning-rate 1e-4 \
  --ppo-ent-coef 1e-3 \
  --steane-action-penalty-coef 0.005 \
  --steane-miscal-penalty-coef 0.001 \
  --steane-n-rounds 6 \
  --steane-shots-per-step 16 \
  --trace-finetune-timesteps 1024 \
  --trace-finetune-rollout-steps 32 \
  --trace-finetune-shots-per-step 16 \
  --trace-finetune-n-rounds 6 \
  --post-eval-episodes 48 \
  --eval-steane-shots-per-step 64 \
  --cycle-sweep-rounds 5,10,15,20,25,30,35,40,45,50 \
  --primary-policy-label full_channel_RL \
  --transfer-policy-label full_channel_transfer_expr1 \
  --transfer-source-checkpoint code/data_generated/rl_steane_tune_experiments_V2/policy_transfer/expr1_source_seed720_checkpoint.pt \
  --save-primary-policy-checkpoint code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/full_channel_rl_showcase_checkpoint.pt \
  --save-json code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseA_showcase_seed1800.json
```

## Notes

- This is not a retraining sweep over `n_rounds`.
- The policy is trained once and then held fixed.
- The transfer checkpoint is also held fixed.
- Only evaluation-time cycle count changes.

## Multi-Seed Batch Execution

For V2, parallel multi-seed execution should be implemented as independent
per-seed commands. This does not change the physical experiment definition; it
only reduces wall-clock time.

Recommended helper:

```bash
PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  -m rl_train.benchmarks.run_expr4_v2_batch \
  --python-exe /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  --seeds 1800,1801,1802 \
  --max-workers 2 \
  --output-dir code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2 \
  --phase-label phaseA_showcase
```

After the runs finish:

```bash
PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  -m rl_train.benchmarks.summarize_expr4_v2_cycle_sweep

PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  -m rl_train.benchmarks.plot_expr4_v2_cycle_sweep
```

## Phase B Focused

If `Phase A showcase` confirms that `full_channel_RL` stays above both
`full_channel_transfer_expr1` and `fixed_zero` over the full cycle sweep,
the next step should be a small robustness check rather than a large new grid.

Recommended `Phase B focused` conditions:

- `B1`: `scale=0.025, f=1e4, g=1.0, p_meas=1e-2`
- `B2`: `scale=0.025, f=1e3, g=1.6, p_meas=1e-2`

Suggested execution:

```bash
PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  -m rl_train.benchmarks.run_expr4_v2_batch \
  --python-exe /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  --seeds 1810,1811,1812 \
  --max-workers 2 \
  --output-dir code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseB_focus_p010_f1e4_g10 \
  --phase-label phaseB_focus_p010_f1e4_g10 \
  --regime-a 0.025 \
  --regime-b 0.25 \
  --corr-f 10000.0 \
  --corr-g 1.0 \
  --p-meas 0.01
```

```bash
PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  -m rl_train.benchmarks.run_expr4_v2_batch \
  --python-exe /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  --seeds 1820,1821,1822 \
  --max-workers 2 \
  --output-dir code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseB_focus_p010_f1e3_g16 \
  --phase-label phaseB_focus_p010_f1e3_g16 \
  --regime-a 0.025 \
  --regime-b 0.25 \
  --corr-f 1000.0 \
  --corr-g 1.6 \
  --p-meas 0.01
```

## Phase C Confirm

`Phase C confirm` should return to the best `Phase A showcase` condition rather
than promote a weaker robustness condition to headline status.

Recommended final confirm condition:

- `C1`: `scale=0.025, f=1e4, g=1.0, p_meas=3e-3`

Recommended execution:

```bash
PYTHONPATH=code /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  -m rl_train.benchmarks.run_expr4_v2_batch \
  --python-exe /Users/wenzhengdong/opt/anaconda3/envs/physics/bin/python \
  --seeds 1830,1831,1832,1833,1834,1835 \
  --max-workers 2 \
  --output-dir code/data_generated/rl_steane_tune_experiments_V2/expr4_cycle_decay_full_composite_v2/phaseC_confirm \
  --phase-label phaseC_confirm \
  --regime-a 0.025 \
  --regime-b 0.25 \
  --corr-f 10000.0 \
  --corr-g 1.0 \
  --p-meas 0.003
```

This confirm stage keeps the same three-way comparison:

- `full_channel_RL`
- `full_channel_transfer_expr1`
- `fixed_zero`

and increases the seed count from `3` to `6`.
