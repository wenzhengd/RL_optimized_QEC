# Composed Correlated `(f, g)` Benchmark Plan

This runbook keeps all produced data inside benchmark-managed paths.

## Output Convention

Use:

- `code/data_generated/benchmarks/composed_corr_fg/phase0_YYYYMMDD/`
- `code/data_generated/benchmarks/composed_corr_fg/phase1_YYYYMMDD/`
- `code/data_generated/benchmarks/composed_corr_fg/phase2_YYYYMMDD/`
- `code/data_generated/benchmarks/composed_corr_fg/phase3_YYYYMMDD/`

Each run writes:

- per-seed JSON files per stage
- top-level `summary.json`

## Phase0 Calibration

```bash
PYTHONPATH=code python -m rl_train.benchmarks.staged_steane_experiments \
  --stages all \
  --stage-specs-json code/rl_train/benchmarks/examples/stage_specs_composed_corr_fg_phase0_calibration.json \
  --seed-workers 5 \
  --device cpu \
  --output-dir code/data_generated/benchmarks/composed_corr_fg/phase0_YYYYMMDD
```

## Phase1 Quick 3x3 Grid

```bash
PYTHONPATH=code python -m rl_train.benchmarks.staged_steane_experiments \
  --stages all \
  --stage-specs-json code/rl_train/benchmarks/examples/stage_specs_composed_corr_fg_phase1_quick_grid.json \
  --seed-workers 5 \
  --device cpu \
  --output-dir code/data_generated/benchmarks/composed_corr_fg/phase1_YYYYMMDD
```

## Phase2 Pilot Focus

```bash
PYTHONPATH=code python -m rl_train.benchmarks.staged_steane_experiments \
  --stages all \
  --stage-specs-json code/rl_train/benchmarks/examples/stage_specs_composed_corr_fg_phase2_pilot_focus.json \
  --seed-workers 5 \
  --device cpu \
  --output-dir code/data_generated/benchmarks/composed_corr_fg/phase2_YYYYMMDD
```

## Phase3 Confirm (Template)

`stage_specs_composed_corr_fg_phase3_confirm_template.json` is a starting point.
After phase2, replace the 3 candidate conditions with the selected winners.

```bash
PYTHONPATH=code python -m rl_train.benchmarks.staged_steane_experiments \
  --stages all \
  --stage-specs-json code/rl_train/benchmarks/examples/stage_specs_composed_corr_fg_phase3_confirm_template.json \
  --seed-workers 5 \
  --device cpu \
  --output-dir code/data_generated/benchmarks/composed_corr_fg/phase3_YYYYMMDD
```

## Summarize One Run

```bash
PYTHONPATH=code python -m rl_train.benchmarks.summarize_composed_fg_grid \
  --summary-json code/data_generated/benchmarks/composed_corr_fg/phase1_YYYYMMDD/summary.json
```

This generates:

- `composed_fg_grid_summary.csv`
- `composed_fg_grid_summary.md`
