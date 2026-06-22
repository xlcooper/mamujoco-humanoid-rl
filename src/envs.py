from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class SingleAgentMaMuJoCoEnv:
    """Small adapter from PettingZoo Parallel API to a single-agent PPO loop."""

    def __init__(
        self,
        domain: str = "Humanoid",
        partitioning: str | None = None,
        seed: int = 0,
        render_mode: str | None = None,
    ) -> None:
        from gymnasium_robotics import mamujoco_v1

        self.env = mamujoco_v1.parallel_env(domain, partitioning, render_mode=render_mode)
        self.seed = seed
        self.agent: str | None = None
        self.observation_space = None
        self.action_space = None

    def reset(self) -> np.ndarray:
        observations, infos = self.env.reset(seed=self.seed)
        self.seed += 1

        if len(self.env.agents) != 1:
            raise ValueError(
                "SingleAgentMaMuJoCoEnv expects exactly one active agent. "
                f"Got agents={self.env.agents}."
            )

        self.agent = self.env.agents[0]
        self.observation_space = self.env.observation_space(self.agent)
        self.action_space = self.env.action_space(self.agent)

        observation = observations[self.agent]
        return observation.astype(np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        if self.agent is None or self.action_space is None:
            raise RuntimeError("Call reset() before step().")

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = {self.agent: clipped_action.astype(np.float32)}

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        reward = float(rewards[self.agent])
        terminated = bool(terminations[self.agent])
        truncated = bool(truncations[self.agent])
        done = terminated or truncated

        if done:
            # PPO handles reset outside this method. The returned observation is unused.
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = observations[self.agent].astype(np.float32)

        info = dict(infos.get(self.agent, {}))
        info["terminated"] = terminated
        info["truncated"] = truncated

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def close(self) -> None:
        self.env.close()


def make_humanoid_single_agent_env(seed: int = 0) -> SingleAgentMaMuJoCoEnv:
    return SingleAgentMaMuJoCoEnv(
        domain="Humanoid",
        partitioning=None,
        seed=seed,
        render_mode=None,
    )
