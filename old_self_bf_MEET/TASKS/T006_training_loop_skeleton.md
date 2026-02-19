# TASKS/T006_training_loop_skeleton.md
Title: Training loop skeleton without policy updates

## Goal
Create a skeleton of a training loop.

## Scope
- src/rl/trainer.py (new)

## Requirements
1) Define:
   training_loop(env, policy, config)
2) Must:
   - iterate epochs and batches
   - call rollout collector
   - *not* perform any actual learning yet
   - log simple metrics (episode lengths, reward sums)

## Acceptance criteria
- The loop runs without errors with dummy policy and environment.
- Outputs metrics without RL updates.
