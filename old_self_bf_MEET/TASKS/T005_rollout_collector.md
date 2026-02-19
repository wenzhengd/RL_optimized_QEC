# TASKS/T005_rollout_collector.md
Title: Rollout collector for environment + policy

## Goal
Collect rollouts of interaction between policy and environment.

## Scope
- src/rl/rollout.py (new)

## Requirements
1) Define function:
   collect_rollouts(policy, env, batch_size, max_steps)
2) Must:
   - run `env.reset()` per trajectory
   - run `policy.act()`
   - call `env.step()`
   - store EpisodeRecord data
3) Must not do RL updates.

## Acceptance criteria
- The function returns a list of EpisodeRecords.
- Each EpisodeRecord includes θ-history + C-history.
