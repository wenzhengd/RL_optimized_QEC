# TASKS/README.md
Incremental Construction Protocol for RL Control Project

This file defines STRICT execution rules for implementing TASK cards.

The goal is controlled, verifiable, minimal-diff development.


---

## 0. Golden Rule

ONE TASK CARD AT A TIME.

Do not:
- combine multiple task cards
- anticipate future tasks
- refactor unrelated modules
- redesign architecture
- add “helpful improvements” outside scope

The project must grow in small, inspectable increments.


---

## 1. Allowed Modification Scope

When implementing a task:

You may ONLY modify files explicitly listed in the task card.

If a file is not listed:
- DO NOT edit it
- DO NOT refactor it
- DO NOT add new dependencies to it

If a task requires additional files:
- The task card must explicitly say so.


---

## 2. Diff Discipline

For each task execution:

- Keep changes minimal.
- Do not change formatting of unrelated code.
- Do not rename variables outside task scope.
- Do not reorganize folder structure.
- Do not auto-optimize.

After completing the task, output:
- A short summary of what was implemented.
- A list of modified files.
- A brief explanation of why changes were necessary.

No long explanations.


---

## 3. No Premature Intelligence

Do NOT:

- implement advanced Monte Carlo schemes
- implement full RL algorithms
- implement real physics
- add extra abstractions “for future safety”
- add logging frameworks
- add configuration frameworks

Unless the current task explicitly requires it.

This project is intentionally staged.


---

## 4. Interface Integrity Rule

Interfaces defined in:

- ARCHITECTURE.md
- PROJECT_CONTEXT.md

are binding.

If a task implementation requires modifying an interface:
- STOP
- Report the conflict
- Do not silently change the interface


---

## 5. Placeholder Philosophy

Stub implementations are acceptable and expected.

Examples:
- simulator may return zeros
- reward may return 0.0
- latent generator may return dummy object
- observation builder may return simple dict

Correct wiring is more important than functionality at this stage.


---

## 6. Future Flexibility Must Be Preserved

All implementations must:

- support high-dimensional continuous θ_i
- support history-dependent observation
- support both step-wise and terminal reward
- support early termination + n_max cap
- remain Monte Carlo extensible

If impl
