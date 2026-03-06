"""Core PPO implementation in PyTorch for the scaffold environment."""

from dataclasses import dataclass
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .config import PPOConfig


class ActorCritic(nn.Module):
    """
    Shared-policy PPO model with Gaussian continuous actions.

    - Actor outputs mean of Normal distribution over theta_t.
    - A learned global log_std parameter controls exploration scale.
    - Critic estimates V(s_t) for advantage/value targets.
    """

    def __init__(self, obs_dim: int, theta_dim: int, hidden_dim: int, use_layer_norm: bool = False) -> None:
        """
        Build actor and critic MLPs.

        Args:
            obs_dim: Observation dimension.
            theta_dim: Action/theta dimension.
            hidden_dim: Width of hidden layers.
            use_layer_norm: If True, apply LayerNorm after hidden linear layers.
        """
        super().__init__()
        def _hidden_block(in_dim: int, out_dim: int) -> nn.Sequential:
            layers = [nn.Linear(in_dim, out_dim)]
            if use_layer_norm:
                # LayerNorm can stabilize optimization when widening the MLP.
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        # Actor: observation -> action mean.
        self.actor = nn.Sequential(
            _hidden_block(obs_dim, hidden_dim),
            _hidden_block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, theta_dim),
        )
        # Critic: observation -> scalar state value.
        self.critic = nn.Sequential(
            _hidden_block(obs_dim, hidden_dim),
            _hidden_block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        # Global log_std shared across states; expanded per batch in get_dist().
        self.log_std = nn.Parameter(torch.zeros(theta_dim))

    def get_dist(self, obs: torch.Tensor) -> Normal:
        """
        Build action distribution pi_theta(.|obs).

        Args:
            obs: Batched observations, shape [B, obs_dim].

        Returns:
            Torch Normal distribution over actions, shape [B, theta_dim].
        """
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate state value.

        Args:
            obs: Batched observations.

        Returns:
            Value predictions shape [B].
        """
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one action and evaluate log_prob/value for rollout collection.

        Args:
            obs: Batched observations.

        Returns:
            action: Sampled action tensor.
            log_prob: Log-probability of sampled action under current policy.
            value: Critic estimate V(obs).
        """
        dist = self.get_dist(obs)
        action = dist.sample()
        # Sum per-dimension log-prob to get a scalar per sample.
        log_prob = dist.log_prob(action).sum(-1)
        value = self.get_value(obs)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate old rollout actions under current policy parameters.

        This is required by PPO to compute importance-sampling ratios and entropy.
        """
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        return log_prob, entropy, value


@dataclass
class Rollout:
    """
    Container for one rollout buffer and its derived training targets.

    All tensors are shape [T, ...] where T = cfg.rollout_steps.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collect_rollout(env, model: ActorCritic, cfg: PPOConfig, obs: np.ndarray, done: bool) -> Tuple[Rollout, np.ndarray, bool]:
    """
    Collect one on-policy rollout and compute GAE advantages/returns.

    Args:
        env: Environment wrapper with reset/step.
        model: Current actor-critic.
        cfg: PPO configuration.
        obs: Current observation at rollout start.
        done: Current done flag at rollout start.

    Returns:
        rollout: Rollout tensors plus computed returns/advantages.
        obs: Latest observation after rollout (for next rollout start).
        done: Latest done flag after rollout.
    """
    device = torch.device(cfg.device)
    obs_buf = []
    action_buf = []
    logprob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []

    for _ in range(cfg.rollout_steps):
        # Forward pass for action sampling and baseline value.
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, logprob_t, value_t = model.get_action_and_value(obs_t)

        # Environment interaction always happens in NumPy space.
        action_np = action_t.squeeze(0).cpu().numpy()
        next_obs, reward, step_done, _ = env.step(action_np)

        # Store transition fields for PPO update later.
        obs_buf.append(obs.copy())
        action_buf.append(action_np.copy())
        logprob_buf.append(float(logprob_t.item()))
        reward_buf.append(float(reward))
        done_buf.append(float(step_done))
        value_buf.append(float(value_t.item()))

        obs = next_obs
        done = bool(step_done)
        if done:
            # Immediately bootstrap next episode so buffer length stays fixed.
            obs = env.reset()
            done = False

    # Bootstrap value for final state of rollout when computing GAE.
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        last_value = float(model.get_value(obs_t).item())

    rewards = np.asarray(reward_buf, dtype=np.float32)
    dones = np.asarray(done_buf, dtype=np.float32)
    values = np.asarray(value_buf, dtype=np.float32)

    # Generalized Advantage Estimation (GAE-Lambda), computed backward in time.
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(cfg.rollout_steps)):
        if t == cfg.rollout_steps - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + cfg.gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    # TD(lambda)-style return target for critic.
    returns = advantages + values

    # Convert rollout arrays into torch tensors once for update efficiency.
    rollout = Rollout(
        obs=torch.tensor(np.asarray(obs_buf, dtype=np.float32), dtype=torch.float32, device=device),
        actions=torch.tensor(np.asarray(action_buf, dtype=np.float32), dtype=torch.float32, device=device),
        log_probs=torch.tensor(np.asarray(logprob_buf, dtype=np.float32), dtype=torch.float32, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.tensor(dones, dtype=torch.float32, device=device),
        values=torch.tensor(values, dtype=torch.float32, device=device),
        returns=torch.tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
    )
    return rollout, obs, done


def _ppo_update(model: ActorCritic, optimizer: optim.Optimizer, rollout: Rollout, cfg: PPOConfig) -> Dict[str, float]:
    """
    Perform PPO optimization on one rollout buffer.

    Implements:
      - clipped policy objective
      - clipped value objective
      - entropy bonus
      - gradient clipping
    """
    n = rollout.obs.shape[0]
    mb = min(cfg.minibatch_size, n)

    # Normalize advantages for stabler policy-gradient scale.
    advantages = rollout.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0
    last_total_loss = 0.0

    for _ in range(cfg.update_epochs):
        # Shuffle full rollout each epoch to randomize minibatches.
        idx = torch.randperm(n, device=rollout.obs.device)
        for start in range(0, n, mb):
            mb_idx = idx[start : start + mb]

            # Evaluate rollout actions under current policy parameters.
            new_log_prob, entropy, new_value = model.evaluate_actions(
                rollout.obs[mb_idx], rollout.actions[mb_idx]
            )

            # Importance-sampling ratio pi_new(a|s) / pi_old(a|s).
            log_ratio = new_log_prob - rollout.log_probs[mb_idx]
            ratio = torch.exp(log_ratio)

            adv = advantages[mb_idx]
            # PPO clipped surrogate objective.
            pg_loss_1 = ratio * adv
            pg_loss_2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

            # Value clipping mirrors common PPO implementations and stabilizes critic.
            value_pred = new_value
            value_old = rollout.values[mb_idx]
            value_clipped = value_old + (value_pred - value_old).clamp(-cfg.clip_eps, cfg.clip_eps)
            value_loss_unclipped = (value_pred - rollout.returns[mb_idx]) ** 2
            value_loss_clipped = (value_clipped - rollout.returns[mb_idx]) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # Entropy term encourages broader action distribution early in training.
            entropy_loss = entropy.mean()
            total_loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

            # Standard optimization step with gradient norm clipping.
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_loss.item())
            last_total_loss = float(total_loss.item())

    # Return last minibatch statistics for lightweight logging.
    return {
        "loss_total": last_total_loss,
        "loss_policy": last_policy_loss,
        "loss_value": last_value_loss,
        "entropy": last_entropy,
        "mean_reward_rollout": float(rollout.rewards.mean().item()),
    }


def train_ppo(
    env,
    cfg: PPOConfig,
    model: Optional[ActorCritic] = None,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[ActorCritic, Dict[str, list]]:
    """
    High-level PPO training loop.

    Args:
        env: Environment wrapper with continuous action interface.
        cfg: PPO configuration dataclass.

    Returns:
        model: Trained ActorCritic model.
        history: Dict of per-update scalar logs.
    """
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Initialize model/optimizer unless a pre-trained model is provided
    # (used by optional phase-2 trace finetuning).
    if model is None:
        model = ActorCritic(
            cfg.obs_dim,
            cfg.theta_dim,
            cfg.hidden_dim,
            use_layer_norm=cfg.use_layer_norm,
        ).to(device)
    else:
        model = model.to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Start from a fresh environment reset.
    obs = env.reset()
    done = False
    num_updates = max(1, cfg.total_timesteps // cfg.rollout_steps)

    history: Dict[str, list] = {
        "loss_total": [],
        "loss_policy": [],
        "loss_value": [],
        "entropy": [],
        "mean_reward_rollout": [],
    }

    for _ in range(num_updates):
        # 1) collect on-policy data
        rollout, obs, done = _collect_rollout(env, model, cfg, obs, done)
        # 2) optimize policy/value on collected data
        stats = _ppo_update(model, optimizer, rollout, cfg)
        # 3) append logs
        for key, value in stats.items():
            history[key].append(value)

    return model, history
