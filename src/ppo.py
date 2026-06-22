from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    observation_dim: int
    action_dim: int
    hidden_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Actor outputs the mean of a diagonal Gaussian policy.
        self.actor_mean = nn.Linear(hidden_size, action_dim)

        # log_std is learned directly and shared across states in this first PPO version.
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic predicts V(s), the expected discounted return from the observation.
        self.critic = nn.Linear(hidden_size, 1)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observations)
        action_mean = self.actor_mean(hidden)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        value = self.critic(hidden).squeeze(-1)
        return action_mean, action_log_std, value

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_log_std, value = self.forward(observations)
        action_std = torch.exp(action_log_std)
        distribution = Normal(action_mean, action_std)

        if actions is None:
            # During rollout we sample from the current stochastic policy.
            actions = distribution.sample()

        # Continuous-control log probabilities are summed across action dimensions.
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)

        return actions, log_prob, entropy, value


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.device = device
        self.step = 0

        self.observations = np.zeros((rollout_steps, observation_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.rewards = np.zeros(rollout_steps, dtype=np.float32)
        self.dones = np.zeros(rollout_steps, dtype=np.float32)
        self.values = np.zeros(rollout_steps, dtype=np.float32)

        self.advantages = np.zeros(rollout_steps, dtype=np.float32)
        self.returns = np.zeros(rollout_steps, dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.observations[self.step] = observation
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = float(done)
        self.values[self.step] = value
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = 0.0

        for index in reversed(range(self.rollout_steps)):
            if index == self.rollout_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[index + 1]

            # done belongs to the transition from s_t to s_{t+1}.
            next_non_terminal = 1.0 - self.dones[index]
            bootstrapped_value = gamma * next_value * next_non_terminal
            td_target = self.rewards[index] + bootstrapped_value
            td_error = td_target - self.values[index]

            # GAE is an exponentially weighted sum of TD errors.
            gae = td_error + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[index] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        indices = np.arange(self.rollout_steps)
        np.random.shuffle(indices)

        for start in range(0, self.rollout_steps, batch_size):
            batch_indices = indices[start : start + batch_size]

            yield {
                "observations": torch.as_tensor(
                    self.observations[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "actions": torch.as_tensor(
                    self.actions[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "old_log_probs": torch.as_tensor(
                    self.log_probs[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "advantages": torch.as_tensor(
                    self.advantages[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "returns": torch.as_tensor(
                    self.returns[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "old_values": torch.as_tensor(
                    self.values[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                ),
            }


def update_ppo(
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
    batch_size: int,
    update_epochs: int,
) -> dict[str, float]:
    losses: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    advantages = buffer.advantages
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    buffer.advantages = normalized_advantages.astype(np.float32)

    for _ in range(update_epochs):
        for batch in buffer.get_batches(batch_size):
            _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                batch["observations"],
                batch["actions"],
            )

            log_ratio = new_log_probs - batch["old_log_probs"]
            ratio = torch.exp(log_ratio)

            # PPO keeps policy updates small by clipping the probability ratio.
            unclipped_policy_loss = -batch["advantages"] * ratio
            clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
            clipped_policy_loss = -batch["advantages"] * clipped_ratio
            policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()

            value_loss = 0.5 * (batch["returns"] - new_values).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss
            loss = loss + config.value_coef * value_loss
            loss = loss - config.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

            losses["policy_loss"].append(float(policy_loss.item()))
            losses["value_loss"].append(float(value_loss.item()))
            losses["entropy"].append(float(entropy_loss.item()))
            losses["approx_kl"].append(float(approx_kl.item()))
            losses["clip_fraction"].append(float(clip_fraction.item()))

    return {name: float(np.mean(values)) for name, values in losses.items()}
