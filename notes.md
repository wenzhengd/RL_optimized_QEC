# Notes (Session Agreement)

Last updated: 2026-02-19

## 1) Project direction
- This is a math/physics research codebase for paper-oriented numerical simulation.
- Keep architecture simple.
- Use only a small number of independent subtasks.

## 2) Final agreed structure (minimal)
- `subtask0_noise/`
- `subtask1_stim_qec/`
- `subtask2_rl/`
- `subtask3_testbed/`
- `data/raw/`
- `data/processed/`
- `results/figures/`
- `results/tables/`

## 3) Subtask definitions
1. `subtask0_noise`: generate sampled noise signals.
2. `subtask1_stim_qec`: Stim-based stabilizer measurement simulation (uses output of subtask0).
3. `subtask2_rl`: neural-network RL scheme (quantum-free internal structure, uses data from subtask1).
4. `subtask3_testbed`: benchmark/test-bed comparing baseline vs RL-improved policy across different QEC codes.

## 4) Data flow
1. subtask0 -> noise files in `data/raw/`
2. subtask1 -> stim measurement files in `data/raw/`
3. subtask2 -> trained policy + metrics in `results/`
4. subtask3 -> benchmark tables/figures in `results/tables/` and `results/figures/`

## 5) Important constraints agreed
- Do not re-introduce heavy/redundant architecture.
- Each subtask should stay very independent and simple.
- Each subtask should have only 1-2 scripts.
- Script files should include comments because the user is learning programming/development.
- Old redundant folders/architecture were requested to be removed.

## 6) Current status
- Minimal 4-subtask folder skeleton has been created.
- Stub scripts with comments were created (skeleton only).
- No advanced implementation has been done yet.

## 7) Next step boundary
- User asked: for now only structure work.
- Next coding step (after user confirms): implement only one target first, likely `subtask0_noise/generate_noise.py`, without touching others.

## 8) How to resume quickly next session
- Open this repo root.
- Read `notes.md` first.
- Then run `ls` and continue from Section 7 above.
