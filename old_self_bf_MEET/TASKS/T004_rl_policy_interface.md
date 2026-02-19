# TASKS/T004_rl_policy_interface.md
Title: RL policy interface stub (no algorithm logic)

## Goal
Define the policy interface for RL with continuous, high-dimensional actions.

## Scope
- src/rl/policy.py (new)

## Requirements
1) Define an abstract policy class or interface.
2) Must accept:
   - observation o_i
   - optional deterministic flag
3) Must return:
   - continuous action θ_i (array-like)
4) Must support future recurrent policies:
   - reset_hidden_state()
   - optional get_hidden_state()

## Deliverables
- policy.py stub with docstrings.

## Acceptance criteria
- The policy interface can be imported.
- policy.act(o_i) returns dummy output (e.g., zeros) with correct shape.
