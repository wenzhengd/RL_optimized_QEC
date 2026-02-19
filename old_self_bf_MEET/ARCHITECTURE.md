# ARCHITECTURE.md (v1)
RL-based high-dimensional continuous control for stochastic latent processes (physics simulator)

## 0. Goals of this document

This document defines:
- the module boundaries (separation of concerns)
- the minimal interfaces between modules
- the episode data flow (what goes where)
- how uncertainty is handled (reward/observation/MC/horizon remain flexible)

This document does NOT define:
- the concrete physics model
- the exact NN architecture
- the exact RL algorithm
- the exact Monte Carlo strategy
Those are deferred to task-specific specs.

---

## 1. System overview

We optimize a control sequence {θ_i} (continuous, high-dimensional) through episodic RL.

An episode consists of:
- a sampled latent-process realization (stochastic, unknown structure)
- iterative control actions θ_i proposed by the agent/policy
- environment simulation producing readouts C_i
- rewards computed step-wise and/or terminal

Key uncertainties intentionally kept open:
- observation content (likely includes control history {θ_≤i})
- reward style (step/terminal/both)
- horizon (fixed max length n + optional early termination)
- Monte Carlo scheme (single trajectory vs multi-trajectory sampling)

Architecture must preserve these degrees of freedom.

---

## 2. Directory / module layout (proposed)

Project root:
- agent.md                     (global constraints for Codex / codegen)
- ARCHITECTURE.md              (this file)
- PROJECT_CONTEXT.md           (physics/math glossary, units, conventions; later)
- TASKS/                       (incremental task cards; later)
- src/
  - env/
    - env_api.md               (interface notes; optional)
    - environment.py           (RL-facing environment wrapper; later)
    - termination.py           (termination logic; later)
  - latent/
    - generator.py             (latent-process generator interface; later)
    - distributions.py         (optional: parameter distributions; later)
  - sim/
    - simulator.py             (physics simulator core interface; later)
    - measurement.py           (readout model; later)
  - obs/
    - builder.py               (observation builder; history-aware; later)
    - encoders.py              (optional: history encoder stubs; later)
  - reward/
    - reward_fn.py             (reward interface; step/terminal; later)
    - objectives.py            (optional: concrete objectives; later)
  - rl/
    - policy.py                (NN policy interface; later)
    - rollout.py               (collect trajectories; later)
    - trainer.py               (training loop skeleton; later)
  - utils/
    - types.py                 (dataclasses / typed dicts: Episode, Step; later)
    - rng.py                   (randomness seeding, streams; later)
    - logging.py               (logging conventions; later)
- tests/                       (added only when requested)

Note:
- This layout is a target; tasks may create these incrementally.
- No module may depend on RL-specific internals unless explicitly allowed.

---

## 3. Core data model (conceptual, not code)

### 3.1 Episode semantics

Episode index: e = 1,2,...

Discrete steps: i = 1...T_e, where
- T_e <= n_max (hard cap)
- episode may terminate early if termination condition triggers

At each step:
- policy proposes action θ_i (continuous, high-dimensional)
- environment runs simulation conditioned on:
  - latent process trajectory (or current latent state)
  - current action θ_i
  - optionally prior actions/history
- environment returns readout C_i and auxiliary diagnostics
- reward may be computed per-step and/or deferred to terminal

### 3.2 Required episode record

The system must be able to record (at minimum):
- actions: {θ_i}
- readouts: {C_i}
- observations returned to the agent: {o_i} (definition flexible)
- termination flags and reasons
- rewards: {r_i} and/or terminal R

Additionally, the environment should optionally record:
- latent trajectory (for debugging/validation only; not exposed to agent unless specified)
- Monte Carlo sampling metadata (e.g., number of samples, seeds)

---

## 4. Interfaces (contracts)

This section defines minimal interfaces between modules.
They should be implemented as Python classes/functions later, but not now.

### 4.1 Latent-process generator interface

Purpose:
- produce stochastic latent realizations for each episode (or each step, if needed)

Contract:
- initialize with RNG/seed control
- sample latent process parameters / trajectory realizations

Minimal capabilities:
- `reset()` for a new episode latent draw
- optional: `step()` to evolve latent state with time

Important: The generator must support future Monte Carlo use:
- ability to sample multiple latent realizations per episode or per batch

### 4.2 Simulator core interface

Purpose:
- physics simulation given latent realization and control input(s)
- generate readouts C_i and auxiliary diagnostics

Contract:
- accept:
  - current control θ_i (and possibly time index i or t_i)
  - access to latent process (trajectory or state)
  - optional: previous controls/history if simulator needs it
- return:
  - readout C_i
  - aux info (diagnostics, intermediate physics quantities)

The simulator must NOT:
- compute reward directly (belongs to reward module)
- decide termination directly (belongs to termination module)
Unless a task explicitly says otherwise.

### 4.3 Observation builder interface (history-aware)

Purpose:
- map raw trajectory info into the observation o_i provided to agent

Design principle:
- observation is intentionally flexible and may include:
  - current readout C_i
  - readout history {C_≤i}
  - control history {θ_≤i} (likely required)
  - time features (i, t_i)
  - extra diagnostics (if permitted)

Contract:
- `build(i, history, latest_readout, aux)` -> observation o_i

Where `history` can include θ and C sequences so far.

Important:
- do NOT assume Markov state
- must support history-dependent observation (POMDP-compatible)
- may later support learned encoders (RNN/Transformer), but not required globally

### 4.4 Reward functional interface (pluggable)

Purpose:
- compute reward signals from trajectory data
- support step-wise and/or terminal reward

Contract:
- must allow:
  - per-step reward r_i (computed during rollout), and/or
  - terminal reward R (computed at end), and/or
  - combined

Reward function should accept:
- trajectory record (θ history, C history, aux history, termination reason)
- optional: Monte Carlo aggregation rule if multiple latent samples used

Reward function must be swappable without changing simulator.

### 4.5 Termination interface (max cap + early stop)

Purpose:
- decide episode termination at each step
- support:
  - hard cap n_max
  - early termination threshold/condition

Contract:
- `check(i, history, latest_readout, aux)` -> (done: bool, reason: str)

The termination module must be configurable:
- thresholds are user-defined
- conditions may depend on C_i, reward, constraints, etc.

### 4.6 Environment wrapper (RL-facing episodic API)

Purpose:
- orchestrate generator + simulator + observation builder + reward + termination
- provide a stable API to RL training code

Environment wrapper responsibilities:
- Episode lifecycle:
  - reset(): initialize new latent realization, clear history, return initial observation
  - step(action θ_i): simulate, update history, compute reward (if step-wise), check termination, return (obs, reward, done, info)
- Support both reward styles:
  - If terminal-only: per-step reward may be 0, terminal computed at end
  - If step-wise: compute each step
  - Combined: both possible

Monte Carlo compatibility:
- environment must support either:
  - single latent realization per episode, OR
  - multiple latent samples aggregated into a single effective readout/reward
This must remain a configuration choice, not hard-coded.

---

## 5. Monte Carlo design options (kept open)

We do NOT choose one now, but interfaces must allow them.

### Option A: Episode-level Monte Carlo (K trajectories per episode)
- For each episode, sample K latent realizations
- For each action θ_i, simulate K readouts C_i^(k)
- Aggregate to produce effective readout \bar{C}_i or effective reward \bar{r}_i

Pros: stable estimates
Cons: expensive

### Option B: Batch-level Monte Carlo
- Each environment instance uses one latent realization
- Monte Carlo average emerges from batch of parallel rollouts

Pros: simple, scalable
Cons: higher variance per trajectory

### Option C: Hybrid / adaptive
- Start with small K, increase if needed
- Use variance reduction later (future task)

The architecture must support A/B without redesign.

---

## 6. Configuration philosophy

All uncertain choices must be configured, not baked into code:
- horizon mode: fixed n_max + early stop enabled/disabled
- observation composition: include θ-history yes/no, include C-history yes/no, etc.
- reward mode: step / terminal / both
- Monte Carlo: single / multi-sample + aggregation rule
- RNG seeding: global seed + per-module streams

Configuration method is not fixed yet (could be dict, dataclass, yaml).
Do not assume one until a task specifies.

---

## 7. Invariants (must always hold)

1. Simulator does not know RL algorithm.
2. Reward logic is pluggable.
3. Observation builder is pluggable and can use history.
4. Termination logic is separate and configurable.
5. Environment wrapper is the only component RL talks to.
6. Small changes should not require cross-module refactors.

---

## 8. Development plan (task-driven)

We will implement incrementally via TASKS/ cards.

Expected order:
1) Define and implement minimal type contracts (Episode/Step records)
2) Implement latent generator interface stub
3) Implement simulator stub returning synthetic C_i
4) Implement observation builder stub (must support θ-history)
5) Implement termination stub (max n + placeholder threshold)
6) Implement reward stub (supports both step/terminal)
7) Implement environment wrapper wiring everything
8) Only then implement RL scheme/training loop

No step should require guessing physics details.

---

END OF ARCHITECTURE.md
