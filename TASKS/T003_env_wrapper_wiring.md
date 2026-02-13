# TASKS/T003_env_wrapper_wiring.md
Title: Implement the RL-facing Environment wrapper wiring all components (still stub physics)

## Goal
Create an Environment wrapper that exposes `reset()` and `step(action)` and orchestrates:
- latent generator
- simulator
- observation builder
- reward functional
- termination logic
while recording EpisodeRecord/StepRecord.

This produces a runnable “minimal closed loop” environment (with stub simulator).

## Scope (files allowed to touch)
- src/env/environment.py          (new)
- src/utils/types.py              (existing; only additive changes allowed)
- src/latent/generator.py         (existing; only additive changes allowed)
- src/sim/simulator.py            (existing; only additive changes allowed)
- src/obs/builder.py              (existing; only additive changes allowed)
- src/reward/reward_fn.py         (existing; only additive changes allowed)
- src/env/termination.py          (existing; only additive changes allowed)

## Requirements

### A) Environment API
Implement:
- reset() -> initial_observation
- step(action θ_i) -> (observation, reward, done, info)

Where:
- action is continuous high-dim (treat as array-like; no dimension assumptions)
- observation returned is produced by ObservationBuilder
- reward returned depends on RewardFunctional configuration:
  - step-wise mode: return r_i each step
  - terminal-only mode: return 0 each step and compute at end
  - combined mode: return r_i plus store terminal R at end (exposed via info)

### B) Episode lifecycle
reset():
- increments episode_id
- calls latent_generator.reset()
- clears history buffers
- returns initial observation:
  - may be empty/zero, or may be built from empty history
  - must be consistent and documented

step(action):
- increments step index
- calls simulator.simulate_step(...)
- appends θ_i, C_i, aux to history
- builds observation o_i (should include θ-history)
- checks termination
- computes reward (as configured)
- records StepRecord
- returns outputs

### C) Early termination
Must support:
- hard cap n_max (always active)
- early stop condition (configurable, can default to off)

When done=True, `info` must include:
- termination reason
- episode summary (length, optional terminal reward)

### D) Monte Carlo compatibility (do not implement MC yet)
The wrapper must be structured so it can later support:
- single latent per episode
- multiple latent samples per episode
without redesign.

Concretely:
- keep a place for MC config and aggregation hooks (even if unused now)
- do not entangle MC assumptions into simulator

## Deliverables
- src/env/environment.py (the wrapper)
- docstrings for reset/step semantics and what is stored in EpisodeRecord
- minimal example usage in comments (not executable tests unless requested)

## Acceptance criteria
- Running a simple loop of:
  - obs = env.reset()
  - for i in range(n_max): obs, r, done, info = env.step(theta_i)
works with stub components.
- EpisodeRecord contains θ-history and C-history of correct length.
- Observation returned includes θ-history (at least θ_≤i) as a component.
- No physics or RL algorithm assumptions appear.
