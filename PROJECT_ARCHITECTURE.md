# RL_QEC_control_tuning Project Architecture

This document is a practical map of the codebase: folder layout, module dependency flow, and execution pipelines.

## 1. Scope

Current active pipeline focuses on:
- Steane-code simulation (`code/quantum_simulation`)
- RL training/evaluation/benchmarking (`code/rl_train`)
- Staged experiment outputs (`code/data_generated`)

Planned extension points already exist for:
- multi-code-family support (`code/rl_train/codes/*`)
- multi-noise-channel support (`code/quantum_simulation/noise_channels.py`)

## 2. Directory Layout (Key Files)

```text
RL_QEC_control_tuning/
├── PROJECT_ARCHITECTURE.md            # this file
├── README.md                          # top-level entry
├── result_analysis.md                 # experiment analysis notes/results
├── resources/                         # papers and reference material
├── code/
│   ├── quantum_simulation/
│   │   ├── steane_code_simulator.py   # Steane circuit simulation + experiment APIs
│   │   ├── noise_engine.py            # core noise model classes (Stim circuit injection)
│   │   ├── noise_channels.py          # noise-channel factory/registry used by RL
│   │   └── sweep_depolarizing_test.py # simulation-only noise sweep script
│   ├── rl_train/
│   │   ├── interfaces.py              # simulator/env interfaces and callback types
│   │   ├── env.py                     # ExternalSimulatorEnv wrapper
│   │   ├── config.py                  # PPOConfig dataclass
│   │   ├── ppo.py                     # PPO implementation
│   │   ├── steane_adapter.py          # Steane simulator adapter (reset/step protocol)
│   │   ├── train.py                   # training entrypoint
│   │   ├── codes/
│   │   │   ├── base.py                # CodeComponents bundle interface
│   │   │   ├── steane.py              # steane-specific builder
│   │   │   └── factory.py             # code-family dispatch
│   │   └── benchmarks/
│   │       ├── eval_steane_ppo.py     # one-shot train+eval benchmark
│   │       ├── staged_steane_experiments.py  # staged batch runner
│   │       ├── sweep_steane_channel_regime.py # channel regime grid sweep
│   │       └── examples/
│   │           └── stage_specs_parametric_regime.json
│   └── data_generated/                # generated benchmark/sweep outputs
└── notes.md
```

## 3. High-Level Dependency Graph

```mermaid
graph TD
    A[noise_engine.py] --> B[noise_channels.py]
    C[steane_code_simulator.py] --> D[steane_adapter.py]
    B --> D

    D --> E[codes/steane.py]
    F[codes/factory.py] --> E
    E --> G[train.py]

    H[env.py] --> G
    I[ppo.py] --> G
    J[config.py] --> I

    G --> K[benchmarks/eval_steane_ppo.py]
    K --> L[benchmarks/staged_steane_experiments.py]
    K --> M[benchmarks/sweep_steane_channel_regime.py]

    L --> N[data_generated/*]
    M --> N
```

## 4. Pipeline: RL Train/Eval Flow

```mermaid
flowchart LR
    U[CLI args] --> T[train.py / eval_steane_ppo.py]
    T --> F[codes.factory: build_code_components]
    F --> S[codes.steane: build_steane_components]
    S --> A[SteaneOnlineSteeringSimulator]
    A --> C[quantum_simulation.steane_code_simulator]
    A --> NC[noise_channels.build_steane_rl_noise_model]
    NC --> NE[noise_engine models]

    S --> E[ExternalSimulatorEnv]
    E --> P[ppo.train_ppo]
    P --> R[trained actor]
    R --> EV[evaluate_steane_policy_fn]
    EV --> OUT[metrics + json report]
```

## 5. Pipeline: Staged Benchmark Flow

```mermaid
flowchart TD
    S0[staged_steane_experiments.py]
    S1[load default or JSON stage specs]
    S2[merge global overrides]
    S3[for each seed: run_benchmark]
    S4[aggregate mean/std metrics]
    S5[write stage seed_*.json + summary.json]

    S0 --> S1 --> S2 --> S3 --> S4 --> S5
```

## 6. Noise Channel Dispatch (Current)

`quantum_simulation/noise_channels.py` is the single dispatch layer used by RL.

Supported channel keys:
- `auto`
- `google_global`
- `google_gate_specific`
- `idle_depolarizing`
- `parametric_google`
- `correlated_pauli_noise_channel`

Selection path:
1. CLI `--steane-noise-channel` from `train.py` / `eval_steane_ppo.py`
2. Stored in `SteaneAdapterConfig.noise_channel`
3. Resolved inside `build_steane_rl_noise_model(...)`
4. Concrete noise model object attached to `SteaneQECSimulator`

## 7. Extension Points (Recommended)

### 7.1 Add New Code Family
1. Add `code/rl_train/codes/<new_code>.py` with builder returning `CodeComponents`.
2. Register in `codes/factory.py`:
   - `available_code_families()`
   - `build_code_components()`
3. Reuse existing `train.py` / benchmark scripts via `--code-family <new_code>`.

### 7.2 Add New Noise Model
1. Implement channel builder in `quantum_simulation/noise_channels.py`.
2. Register key in:
   - `SteaneNoiseChannel`
   - `available_steane_noise_channels()`
   - `build_steane_rl_noise_model(...)`
3. Expose key in CLI choices in:
   - `rl_train/train.py`
   - `rl_train/benchmarks/eval_steane_ppo.py`

### 7.3 Replace Correlated Channel Physics
Edit only:
- `correlated_pauli_model_kernel(...)` in `quantum_simulation/noise_channels.py`

Keep contract:
- return `p_total(q,t)` in `[0, p_clip_max]`.

## 8. Data/Output Conventions

- Single benchmark run: JSON report from `eval_steane_ppo.py`
- Staged runs: per-stage folders with `seed_<id>.json` and global `summary.json`
- Sweep runs: grid-style JSON summaries

Primary generated root:
- `code/data_generated/`

## 9. Fast Orientation (for collaborators)

If you only read 5 files, read in this order:
1. `code/rl_train/benchmarks/eval_steane_ppo.py`
2. `code/rl_train/steane_adapter.py`
3. `code/quantum_simulation/noise_channels.py`
4. `code/quantum_simulation/steane_code_simulator.py`
5. `code/rl_train/benchmarks/staged_steane_experiments.py`

## 10. Command Entry Call Graph

### 10.1 Main command-to-module map

```mermaid
flowchart TD
    C1[python -m rl_train.train] --> T1[train.py::main]
    T1 --> T2[codes.factory::build_code_components]
    T2 --> T3[steane_adapter::SteaneOnlineSteeringSimulator]
    T1 --> T4[env::ExternalSimulatorEnv]
    T4 --> T5[ppo::train_ppo]
    T5 --> T6[optional post-eval]

    C2[python -m rl_train.benchmarks.eval_steane_ppo] --> E1[eval_steane_ppo::run_benchmark]
    E1 --> E2[build Steane components]
    E1 --> E3[ppo::train_ppo]
    E1 --> E4[evaluate_steane_policy_fn]
    E1 --> E5[save benchmark JSON optional]

    C3[python -m rl_train.benchmarks.staged_steane_experiments] --> S1[load stage specs default/JSON]
    S1 --> S2[for each seed call run_benchmark]
    S2 --> E1
    S2 --> S3[aggregate metrics]
    S3 --> S4[write seed_*.json + summary.json]

    C4[python -m rl_train.benchmarks.sweep_steane_channel_regime] --> W1[parse a,b grid]
    W1 --> W2[for each grid point call run_benchmark]
    W2 --> E1
    W2 --> W3[write sweep JSON]

    C5[python -m quantum_simulation.sweep_depolarizing_test] --> Q1[build idle depolarizing noise]
    Q1 --> Q2[SteaneQECSimulator run trace]
    Q2 --> Q3[plot PDF + sweep metrics]
```

### 10.2 Runtime stack under `run_benchmark`

```mermaid
flowchart LR
    RB[eval_steane_ppo.run_benchmark] --> BC[_build_steane_components]
    BC --> CF[codes.factory]
    CF --> CS[codes.steane]
    CS --> SA[SteaneOnlineSteeringSimulator]
    SA --> NC[noise_channels.build_steane_rl_noise_model]
    NC --> NE[noise_engine noise classes]
    SA --> SQ[SteaneQECSimulator]
    RB --> PPO[ppo.train_ppo]
    RB --> EV[_evaluate_policies]
```

### 10.3 Output artifact mapping by command

- `train.py`:
  - terminal logs; optional post-eval summary in stdout
- `eval_steane_ppo.py`:
  - optional one-run report JSON via `--save-json`
- `staged_steane_experiments.py`:
  - per-stage `seed_<id>.json`
  - stage-group `summary.json`
- `sweep_steane_channel_regime.py`:
  - one sweep JSON report (grid + aggregate + runs)
- `sweep_depolarizing_test.py`:
  - sweep metrics and generated plot files
