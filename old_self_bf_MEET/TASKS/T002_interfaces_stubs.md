# TASKS/T002_interfaces_stubs.md
Title: Define minimal interfaces for generator/simulator/obs/reward/termination (stubs only)

## Goal
Create minimal class/function interfaces (with placeholders) so modules can be wired together.
Do NOT implement real physics or RL logic yet.

## Scope (files allowed to touch)
- src/latent/generator.py        (new)
- src/sim/simulator.py           (new)
- src/obs/builder.py             (new)
- src/reward/reward_fn.py        (new)
- src/env/termination.py         (new)
- src/utils/types.py             (existing from T001; only additive changes allowed)

## Requirements

### A) Latent generator interface
Must support:
- reset() -> latent_state_or_trajectory (opaque object)
- optional step(i, ...) if needed later (can be placeholder)

Must include RNG/seed handling hooks (no policy assumed).

### B) Simulator interface
Must define a function/class method like:
- simulate_step(i, action, latent, history, rng) -> (readout, aux)

Stub behavior:
- Return deterministic synthetic output (e.g., zeros) unless randomness is explicitly required.
- Must not compute reward or termination.

### C) Observation builder interface (history-aware)
Must define:
- build(i, history, latest_readout, aux) -> observation

Stub behavior:
- Default observation should include θ-history up to i (or at least latest θ_i) in a simple structure.
- Keep it generic (dict recommended).

### D) Reward functional interface (pluggable)
Must support step/terminal/both without committing:
- compute_step(i, history, latest_readout, aux) -> r_i (optional)
- compute_terminal(episode_record) -> R (optional)

Stub behavior:
- Return 0.0 by default.

### E) Termination interface (n_max + early stop)
Must define:
- check(i, history, latest_readout, aux) -> (done, reason)

Stub behavior:
- done if i >= n_max
- early stop hook exists but can be disabled by default (e.g., threshold=None)

## Deliverables
- New modules containing interfaces + minimal stub implementations.
- Clear docstrings for each interface.

## Acceptance criteria
- All modules can be imported without circular imports.
- Interfaces are generic and do not assume:
  - finite vs infinite horizon (beyond episodic)
  - specific MC scheme
  - specific reward style
  - specific observation structure
- The observation builder stub supports inclusion of θ-history.
