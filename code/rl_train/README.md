# PyTorch PPO Scaffold (Simulator-Callable)

This folder is a minimal scaffold to build your own RL control task in the same coding spirit as this repo:

- time is discrete (`t_1 -> ... -> t_N`)
- policy outputs parameters `theta_t`
- external mapper transforms `theta_t -> action_t`
- simulator runs `action_t` and returns feedback
- reward function computes `r_t`
- PPO updates policy/value networks

## Files

- `config.py`: user-facing hyperparameters (`max_steps` included).
- `interfaces.py`: simulator and callback interfaces.
- `env.py`: environment wrapper (`action_mapper`, `reward_fn`, `terminate_fn`).
- `ppo.py`: PyTorch actor-critic PPO trainer.
- `example_simulator.py`: runnable toy simulator.
- `train.py`: entry script with TODO hooks for your real simulator and reward.

## Install

This scaffold uses PyTorch and NumPy:

```bash
pip install torch numpy
```

## Run the sanity check

```bash
python -m torch_ppo_scaffold.train
```

## Plug in your task

In `train.py`, replace:

1. `YourSimulator` with your callable simulator.
2. `todo_reward_fn` with your reward formula.
3. `identity_action_mapper` with your external `a(theta)` mapping.
4. `obs_dim`, `theta_dim`, `max_steps` in `PPOConfig`.

