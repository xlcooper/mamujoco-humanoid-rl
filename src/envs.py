from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None


@dataclass
class StepResult:
    """把环境一步返回值整理成 PPO 训练更好读的结构。"""

    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class SingleAgentMaMuJoCoEnv:
    """把 MaMuJoCo 的 PettingZoo Parallel API 适配成单智能体 PPO 接口。"""

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
        # MaMuJoCo 返回 dict：{agent_name: observation}。
        # partitioning=None 时只有 agent_0，所以这里转成单个 observation。
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

        # 高斯策略采样出来的动作可能略超环境边界，送入 MuJoCo 前先裁剪。
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = {self.agent: clipped_action.astype(np.float32)}

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        reward = float(rewards[self.agent])
        terminated = bool(terminations[self.agent])
        truncated = bool(truncations[self.agent])
        done = terminated or truncated

        if done:
            # done 后训练主循环会 reset，这里的 observation 只是占位。
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
    # Stage 1 的普通 PPO baseline 使用单智能体 Humanoid。
    return SingleAgentMaMuJoCoEnv(
        domain="Humanoid",
        partitioning=None,
        seed=seed,
        render_mode=None,
    )


class GymnasiumSingleAgentMaMuJoCoEnv(gym.Env if gym is not None else object):
    """把 MaMuJoCo 单智能体 ParallelEnv 包成 SB3 需要的 Gymnasium Env。"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        domain: str = "Humanoid",
        partitioning: str | None = None,
        seed: int = 0,
        render_mode: str | None = None,
    ) -> None:
        if gym is None:
            raise ModuleNotFoundError(
                "gymnasium is required for GymnasiumSingleAgentMaMuJoCoEnv. "
                "Install project requirements before running SB3 SAC."
            )

        from gymnasium_robotics import mamujoco_v1

        self.env = mamujoco_v1.parallel_env(domain, partitioning, render_mode=render_mode)
        self.next_seed = seed
        self.agent = self._discover_single_agent()
        self.observation_space = self.env.observation_space(self.agent)
        self.action_space = self.env.action_space(self.agent)
        self.render_mode = render_mode

    def _discover_single_agent(self) -> str:
        possible_agents = list(getattr(self.env, "possible_agents", []))
        if len(possible_agents) != 1:
            raise ValueError(
                "GymnasiumSingleAgentMaMuJoCoEnv expects exactly one possible agent. "
                f"Got possible_agents={possible_agents}."
            )
        return possible_agents[0]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options

        actual_seed = self.next_seed if seed is None else seed
        super().reset(seed=actual_seed)
        observations, infos = self.env.reset(seed=actual_seed)
        self.next_seed = actual_seed + 1

        if len(self.env.agents) != 1:
            raise ValueError(
                "GymnasiumSingleAgentMaMuJoCoEnv expects exactly one active agent. "
                f"Got agents={self.env.agents}."
            )

        self.agent = self.env.agents[0]
        observation = observations[self.agent].astype(np.float32)
        info = dict(infos.get(self.agent, {}))
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # SB3 的 SAC 已经使用 squashed action，但这里仍裁剪一次，避免数值边界误差。
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = {self.agent: clipped_action.astype(np.float32)}

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        reward = float(rewards[self.agent])
        terminated = bool(terminations[self.agent])
        truncated = bool(truncations[self.agent])

        if self.agent in observations:
            observation = observations[self.agent].astype(np.float32)
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = dict(infos.get(self.agent, {}))
        info["terminated"] = terminated
        info["truncated"] = truncated

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self.env.render()

    def close(self) -> None:
        self.env.close()


def make_humanoid_gymnasium_env(
    seed: int = 0,
    render_mode: str | None = None,
) -> GymnasiumSingleAgentMaMuJoCoEnv:
    # Stage 3 的 SB3 SAC baseline 继续使用 partitioning=None 的单智能体 Humanoid。
    return GymnasiumSingleAgentMaMuJoCoEnv(
        domain="Humanoid",
        partitioning=None,
        seed=seed,
        render_mode=render_mode,
    )
