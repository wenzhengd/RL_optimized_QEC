"""
PyTorch PPO scaffold for simulator-driven control tasks.

This package is intentionally small and split by responsibility:
  - config: PPO hyperparameters and run options.
  - interfaces: callable simulator contract and callback signatures.
  - env: environment wrapper that binds policy outputs to simulator actions.
  - ppo: actor-critic PPO training loop.
  - train: runnable entry-point with TODO hooks for customization.
"""
