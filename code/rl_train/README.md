# PyTorch PPO Scaffold (Simulator-Callable)

This folder is a minimal scaffold to build your own RL control task in the same coding spirit as this repo:

- time is discrete (`t_1 -> ... -> t_N`)
- policy outputs parameters `theta_t`
- external mapper transforms `theta_t -> action_t`
- simulator runs `action_t` and returns feedback
- reward function computes `r_t`
- PPO updates policy/value networks

## Files

- `config.py`: user-facing hyperparameters (`max_steps` included).
- `interfaces.py`: simulator and callback interfaces.
- `env.py`: environment wrapper (`action_mapper`, `reward_fn`, `terminate_fn`).
- `ppo.py`: PyTorch actor-critic PPO trainer.
- `example_simulator.py`: runnable toy simulator.
- `steane_adapter.py`: Steane-QEC adapter (`reset/step`) and Google-like drifted gate-noise bridge.
- `train.py`: entry script with TODO hooks for your real simulator and reward.

## Install

This scaffold uses PyTorch and NumPy:

```bash
pip install torch numpy
```

## Run

```bash
python -m rl_train.train --backend toy --total-timesteps 5000
```

Run Steane-backed training (default backend):

```bash
python -m rl_train.train --backend steane
```

Run PPO with paper-inspired settings (optimizer remains PPO):

```bash
python -m rl_train.train --backend steane --google-paper-ppo-preset
```

Run with post-training learned-vs-fixed DR/LER report:

```bash
python -m rl_train.train \
  --backend steane \
  --google-paper-ppo-preset \
  --post-eval-episodes 30
```

Run dedicated benchmark script (includes random baseline and optional JSON output):

```bash
python -m rl_train.benchmarks.eval_steane_ppo \
  --google-paper-ppo-preset \
  --post-eval-episodes 30 \
  --eval-steane-shots-per-step 128 \
  --save-json code/data_generated/steane_ppo_benchmark.json
```

Run fast-train + trace-finetune benchmark (phase-2 keeps PPO optimizer unchanged):

```bash
python -m rl_train.benchmarks.eval_steane_ppo \
  --total-timesteps 512 \
  --rollout-steps 32 \
  --trace-finetune-timesteps 64 \
  --trace-finetune-rollout-steps 16 \
  --trace-finetune-shots-per-step 8 \
  --post-eval-episodes 32 \
  --eval-steane-shots-per-step 64
```

Run the same benchmark with a wider LayerNorm MLP policy/value network:

```bash
python -m rl_train.benchmarks.eval_steane_ppo \
  --total-timesteps 512 \
  --rollout-steps 32 \
  --ppo-hidden-dim 256 \
  --ppo-use-layer-norm \
  --trace-finetune-timesteps 64 \
  --trace-finetune-rollout-steps 16 \
  --trace-finetune-shots-per-step 8 \
  --post-eval-episodes 32 \
  --eval-steane-shots-per-step 64
```

Run the staged protocol (sanity -> pilot -> scale, optional power stage):

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --seed-workers 5 \
  --output-dir code/data_generated/steane_staged_runs
```

Apply a global code/noise override to all selected stages:

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --stages 1,4,6 \
  --code-family steane \
  --steane-noise-channel parametric_google \
  --seed-workers 5 \
  --output-dir code/data_generated/steane_staged_runs_parametric
```

Use a JSON file for global staged overrides (forwarded to `eval_steane_ppo` args):

```json
{
  "code_family": "steane",
  "steane_noise_channel": "parametric_google",
  "steane_channel_regime_a": 1.2,
  "steane_channel_regime_b": 0.8
}
```

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --stages 8,9,10 \
  --base-overrides-json path/to/stage_overrides.json \
  --seed-workers 5 \
  --output-dir code/data_generated/steane_staged_runs_regime
```

Use a custom stage-spec JSON (config-driven staged runs, no Python edits):

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --stage-specs-json code/rl_train/benchmarks/examples/stage_specs_parametric_regime.json \
  --stages all \
  --seed-workers 2 \
  --output-dir code/data_generated/steane_custom_stage_specs
```

Run the added power-focused stage only:

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --stages 4 \
  --seed-workers 5 \
  --output-dir code/data_generated/steane_staged_runs_power
```

Run the trace-finetune stage:

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --stages 6 \
  --seed-workers 5 \
  --output-dir code/data_generated/steane_staged_runs_trace_finetune
```

Run architecture fairness stage (same budget as stage8, wider LayerNorm MLP):

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --stages 11 \
  --seed-workers 5 \
  --output-dir code/data_generated/steane_staged_runs_arch_mlp
```

Run wider+LayerNorm MLP hyperparameter tuning sweep (A/B/C, 5 seeds each):

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --stages 12,13,14 \
  --seed-workers 5 \
  --output-dir code/data_generated/steane_staged_runs_arch_tune
```

Run tuned wider+LayerNorm MLP confirmation stage (10 seeds):

```bash
python -m rl_train.benchmarks.staged_steane_experiments \
  --stages 15 \
  --seed-workers 5 \
  --output-dir code/data_generated/steane_staged_runs_arch_tuned_confirm
```

Sweep RL performance over correlated-channel parameters `(f,g)`:

```bash
python -m rl_train.benchmarks.sweep_steane_channel_regime \
  --corr-f-values 1e3,1e4,1e5 \
  --corr-g-values 0.6,1.0,1.4 \
  --force-channel correlated_pauli_noise_channel \
  --output-json code/data_generated/steane_channel_regime_sweep.json \
  --total-timesteps 512 \
  --rollout-steps 32 \
  --trace-finetune-timesteps 64 \
  --trace-finetune-rollout-steps 16 \
  --trace-finetune-shots-per-step 8 \
  --post-eval-episodes 16 \
  --eval-steane-shots-per-step 32
```

Run legacy global-control baseline (all gates share one global control vector):

```bash
python -m rl_train.train --backend steane --steane-control-mode global
```

Run round-wise online mode (one RL step = one QEC round):

```bash
python -m rl_train.train \
  --backend steane \
  --steane-stepping-mode online_rounds \
  --steane-n-rounds 10 \
  --max-steps 10
```

Reset drift phase at every episode (optional):

```bash
python -m rl_train.train --backend steane --steane-reset-drift-on-episode
```

## Plug in your task

In `train.py`, replace:

1. `YourSimulator` with your callable simulator.
2. `steane_reward_fn` / `make_steane_reward_fn` (or `example_reward_fn`) with your reward formula.
3. `identity_action_mapper` with your external `a(theta)` mapping.
4. `obs_dim`, `theta_dim`, `max_steps` in `PPOConfig`.

## Steane Adapter Notes

- File: `steane_adapter.py`
- Adapter exposes `reset()/step()` so it can be used by `ExternalSimulatorEnv`.
- Default Steane PPO setup in `train.py` is paper-inspired:
  - `steane-control-mode=gate_specific`
  - `steane-n-1q-control-slots=24`
  - `steane-n-2q-control-slots=24`
  - `steane-stepping-mode=candidate_eval`
  - `steane-n-rounds=25`
  - `rollout-steps=40`
  - `max-steps=1`
  - reward mode `paper_surrogate` (maximize negative detector rate proxy)
- Control modes:
  - `gate_specific` (default): gate instruction maps to dedicated control slot.
  - `global`: shared control mismatch drives one global `p_1q/p_2q`.
- Noise-channel layer (new):
  - `--steane-noise-channel auto` (default): legacy mapping from control mode.
  - `--steane-noise-channel google_gate_specific` / `google_global`: Google-like gate depolarizing channels.
  - `--steane-noise-channel idle_depolarizing`: action-independent idle Pauli channel.
  - `--steane-noise-channel parametric_google`: Google-like channel with regime knobs
    `--steane-channel-regime-a` and `--steane-channel-regime-b`.
  - `--steane-noise-channel correlated_pauli_noise_channel`: temporally correlated
    idle Pauli channel with explicit `(f,g)` controls and direction/qubit independence.
  - `--steane-noise-channel composed_google_gate_specific_correlated` /
    `composed_google_global_correlated`: one-pass composed model that injects
    Google-like gate depolarizing noise and correlated idle Pauli noise together.
  - idle channel parameters:
    `--steane-idle-p-total-per-idle`, `--steane-idle-px-weight`, `--steane-idle-py-weight`,
    `--steane-idle-pz-weight`.
  - correlated channel parameters:
    `--steane-channel-corr-f` (Hz, lower means slower drift / longer memory),
    `--steane-channel-corr-g` (overall channel-strength scale),
    `--steane-channel-corr-g-mode` (`per_window` or `per_circuit`).
    Use `per_circuit` when comparing different circuit lengths and you want
    g to represent a length-normalized circuit-level budget.

Quick smoke example for the custom correlated channel:

```bash
python -m rl_train.benchmarks.eval_steane_ppo \
  --total-timesteps 8 \
  --rollout-steps 2 \
  --steane-n-rounds 1 \
  --steane-shots-per-step 2 \
  --post-eval-episodes 1 \
  --eval-steane-shots-per-step 2 \
  --steane-noise-channel composed_google_gate_specific_correlated \
  --steane-channel-corr-f 1e4 \
  --steane-channel-corr-g 1.0 \
  --steane-channel-corr-g-mode per_circuit
```

Where to plug in your own correlated physics model:

- Edit this builder:
  [code/quantum_simulation/noise_channels.py](/Users/wenzheng/Desktop/RL_QEC_control_tuning/code/quantum_simulation/noise_channels.py)
  `build_correlated_pauli_noise_channel(...)`
- The builder creates a two-state Hidden-Markov telegraph Pauli channel with:
  - explicit correlation frequency `f` and strength scale `g`
  - qubit-independent and direction-independent dynamics
  - independent X/Y/Z hidden chains sharing the same `(f,g)` parameters.
- Actual stateful injection logic lives in:
  [code/quantum_simulation/noise_engine.py](/Users/wenzheng/Desktop/RL_QEC_control_tuning/code/quantum_simulation/noise_engine.py)
  `HiddenMarkovCorrelatedPauliNoiseModel`.
- One-pass gate+idle composition lives in the same file as:
  `ComposedGateAndCorrelatedIdleNoiseModel`.
- For correctness, this channel runs with effective `shot_workers=1`
  (state is reset at each shot and evolves across idle windows inside that shot).
- Two stepping modes:
  - `candidate_eval`: each RL step runs a full `n_rounds` memory experiment.
  - `online_rounds`: each RL step runs exactly 1 round; simulator returns `done=True` after `n_rounds` steps.
- Reward modes:
  - `paper_surrogate` (default): reward from detector/stabilizer event-rate surrogate.
  - `legacy_success`: original success-rate shaping reward.
- Oracle leakage control:
  - by default, oracle-only metric `miscalibration_mse` is hidden from observation/info.
  - set `--steane-expose-oracle-metrics` only for debugging/ablation.
  - `--steane-miscal-penalty-coef` default is `0.0` to avoid oracle reward leakage.
- Trace collection / runtime:
  - default uses fast summary mode (`collect_traces=False`) for RL training/eval.
  - set `--steane-collect-traces` only when you explicitly need per-shot trace diagnostics.
  - trace mode is much slower because it stores full shot histories.
  - `--steane-shot-workers N` enables shot-level simulator parallelism in fast summary mode.
  - trace mode currently keeps single-thread behavior for reproducibility.
- Post-train evaluation:
  - `--post-eval-episodes N` compares learned PPO policy with fixed-zero policy.
  - `--eval-steane-shots-per-step K` (K>0) raises evaluation shot count only, to reduce metric noise
    without increasing training simulation cost.
  - `--trace-eval-episodes M` enables an additional trace-based evaluation pass for higher-fidelity detector metrics.
  - `--trace-eval-steane-shots-per-step K` controls shot count in that trace-based evaluation pass.
  - evaluation uses separate simulators with drift reset per episode to avoid phase-drift comparison bias.
  - reports detector-rate (DR) and LER proxy (`1 - success_rate`).
- Optional phase-2 trace finetune (PPO unchanged):
  - set `--trace-finetune-timesteps > 0` to enable a second training phase with `collect_traces=True`.
  - use `--trace-finetune-rollout-steps`, `--trace-finetune-shots-per-step`, and
    `--trace-finetune-n-rounds` to control phase-2 compute budget.
- Drift time:
  - default keeps drift phase continuous across episodes.
  - set `--steane-reset-drift-on-episode` to restart drift phase at episode reset.
- Mapping for Steane:
  - `n_stab = 6`
  - `n_steps = 6 * n_rounds`
