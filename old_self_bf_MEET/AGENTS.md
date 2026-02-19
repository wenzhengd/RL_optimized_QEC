# RL Control Project — Agent Specification (v3)

## ROLE

You are a structured code-generation assistant working inside a modular scientific project.

This project implements:
- Reinforcement Learning (RL), episodic setting
- Neural Network (NN) policy
- Physics-based / simulation-based environment dynamics
- High-dimensional continuous control optimization

Your task is to generate code incrementally and modularly.
You MUST follow the rules below.


---

## PROJECT CORE IDEA (ABSTRACT, NON-COMMITTAL)

We design a control sequence {θ_i} using an RL neural-network policy.

Environment contains:
- A latent/hidden process (NOT assumed to be a single scalar f(t); it may be high-dimensional, structured, and stochastic)
- A simulation pipeline (physics + measurement/readout)

At discrete times t_i, the simulator returns a readout C_i.
The observed history is:
    H_i = {C_1, C_2, ..., C_i}

The RL objective is to learn a policy that generates {θ_i} to optimize a user-defined reward functional based on the trajectory outcomes.


---

## EPISODIC RL SETTING & TERMINATION

Training is episodic.

Episodes must support BOTH:
- a fixed maximum length n (hard cap), and
- early termination triggered by a user-defined threshold/condition.

Do NOT assume finite horizon vs infinite horizon globally. The design must remain compatible with:
- finite-horizon episodic tasks (max length n)
- variable-length episodes (early termination)
- horizon-free formulations (termination may be stochastic or condition-based)


---

## ACTION SPACE (CONTROL)

At each time step i:
- Action θ_i is CONTINUOUS and HIGH-DIMENSIONAL.
- For fixed i, θ_i contains multiple continuous parameters to be optimized.
- Do NOT discretize θ_i unless explicitly required by a task spec.


---

## LATENT PROCESS & STOCHASTICITY

The hidden/latent process is UNKNOWN and STOCHASTIC.

There MUST be a signal / latent-process generator responsible for producing latent trajectories.

Environment simulation must support Monte Carlo style sampling:
- multiple latent trajectories per training batch if required
- stochastic readout generation conditioned on latent trajectory and actions

Do NOT expand the detailed physics of the latent process in global documents unless explicitly requested.


---

## OBSERVATIONS (LIKELY HISTORY-DEPENDENT)

Observation design is intentionally flexible.

However, it is LIKELY that observations will include control history:
    {θ_1, θ_2, ..., θ_i}
(and possibly readout history {C_1,...,C_i} and other diagnostics).

Therefore:
- The environment/RL interface MUST support optionally providing history-dependent observations.
- Do NOT assume full observability or Markov property globally.
- The project must remain compatible with:
  - Markov observations (state-based)
  - history-based / POMDP settings (partial observability)
  - recurrent policies or explicit history encoders if later specified


---

## REWARD DESIGN (GENERALITY REQUIRED)

Reward functional is currently unspecified and may evolve.

Reward may be:
- step-wise: r_i computed at each time step, and/or
- terminal-only: R computed after episode ends, and/or
- combined

Therefore:
- Do NOT hard-code one reward style globally.
- Reward computation MUST be modular/pluggable, defined task-by-task.
- Prefer interfaces that accept full trajectory data and return reward(s).


---

## NON-NEGOTIABLE RULES

1. DO NOT generate diagrams.
2. DO NOT implement everything in one step.
3. DO NOT collapse modules into one file.
4. DO NOT assume hidden physics details.
5. DO NOT invent undocumented functionality.
6. NEVER refactor multiple modules unless explicitly requested.
7. Prefer minimal changes per task; keep diffs small and reviewable.


---

## MODULARITY PRINCIPLE

The system must remain separated into:
- Environment interface (RL-facing episodic API)
- Dynamics simulator core (physics/measurement, pluggable)
- Latent-process / signal generator (stochastic source)
- Observation builder (can include history θ_{≤i}, C_{≤i})
- Reward functional (pluggable: step-wise/terminal/both)
- RL scheme / policy model
- Training loop
- Validation/evaluation

Each module must have:
- clear interface
- minimal dependency
- no circular imports


---

## STRICT OUTPUT POLICY (FOR CODE GENERATION TASKS)

When implementing a task:
- Only modify files explicitly mentioned in the task file.
- Do not create extra files unless requested.
- Do not change unrelated modules.
- Always respect the interface definition.
- Include minimal tests/checks only if the task requires them.


---

## DEVELOPMENT STRATEGY (ORDERED, INCREMENTAL)

Build in this order unless a task says otherwise:
1. Define environment interface (episodic API, termination hooks)
2. Define latent-process generator interface (sampling protocol)
3. Define observation builder interface (history-aware, optional)
4. Implement simulator placeholder with deterministic stubs
5. Connect Monte Carlo sampling pathway (latent trajectories → readouts)
6. Define RL interface (policy, rollout collector)
7. Implement training loop (configurable)
8. Implement reward functional(s) (pluggable)
9. Validation & evaluation harness

Never jump ahead.


---

## SUCCESS CRITERION

The final system must:
- Learn a policy generating continuous high-dimensional control sequence {θ_i}
- Support history-dependent observations including {θ_1,...,θ_i} (optionally)
- Interact with a simulator producing readouts {C_i} per episode
- Support stochastic latent-process generation + Monte Carlo sampling
- Support fixed max length n + early termination thresholds
- Optimize a user-defined objective (reward functional, pluggable)
- Return best learned policy
- Support validation/evaluation

END OF AGENT SPECIFICATION
