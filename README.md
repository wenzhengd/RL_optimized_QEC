# RL_QEC_control_tuning (Minimal Structure)

This repo is organized into 4 independent subtasks for paper-oriented numerical simulation.

## Subtasks
- `subtask0_noise/`: generate sampled noise signals
- `subtask1_stim_qec/`: run Stim-based stabilizer measurement simulation
- `subtask2_rl/`: train/evaluate RL policy (quantum-free internal structure)
- `subtask3_testbed/`: benchmark and compare policies across QEC codes

## Data flow
1. subtask0 -> noise file in `data/raw/`
2. subtask1 -> measurement file in `data/raw/`
3. subtask2 -> policy/metrics in `results/`
4. subtask3 -> tables/figures in `results/`
