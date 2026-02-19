# TASKS/T001_types_and_records.md
Title: Define trajectory record types (Episode/Step) without committing to physics details

## Goal
Create minimal, stable data contracts for storing an episode rollout.
These types will be shared across env/sim/obs/reward/rl modules.

## Scope (files allowed to touch)
- src/utils/types.py   (new)
- (optional) src/utils/__init__.py (new, only if needed)

## Requirements
1) Define conceptual records for:
   - StepRecord
   - EpisodeRecord

2) StepRecord must minimally support fields (names may vary but meaning must match):
   - step_index i (int)
   - action θ_i (continuous high-dim; store as generic array-like)
   - readout C_i (generic scalar or vector; store as generic array-like)
   - observation o_i (generic; can be array-like or dict)
   - reward r_i (float, optional depending on reward mode)
   - done flag (bool)
   - info / aux (dict-like for diagnostics)

3) EpisodeRecord must minimally support:
   - episode_id (int or str)
   - n_max (int) and actual length T (int)
   - sequences of actions, readouts, observations, rewards (list-like)
   - termination reason (str)
   - optional: latent metadata (dict) for debugging only
   - optional: Monte Carlo metadata (dict) for debugging only

4) Types must not assume:
   - C_i scalar vs vector
   - θ_i dimensionality
   - Markov observation
   - step-wise vs terminal reward

5) Keep the types lightweight and serializable.
   Prefer dataclasses or TypedDict.
   Do not import heavy ML frameworks in types.py.

## Deliverables
- src/utils/types.py defining the records.
- Inline docstrings explaining each field meaning.

## Acceptance criteria
- The records can represent:
  - step-wise rewards and/or terminal-only reward
  - early termination before n_max
  - history-dependent observations containing θ_≤i
- No physics/RL algorithm assumptions appear in the code.
