from __future__ import annotations

import argparse
from typing import Any


def parse_partitioning(raw_value: str) -> str | None:
    # 命令行里用 "none" 表示单智能体 Humanoid，对应 MaMuJoCo 的 partitioning=None。
    normalized = raw_value.strip().lower()
    if normalized in {"none", "null", "single", "single-agent"}:
        return None
    return raw_value


def summarize_space(space: Any) -> str:
    # 冒烟测试只需要看空间类型、shape 和 dtype，避免打印过长内容。
    shape = getattr(space, "shape", None)
    dtype = getattr(space, "dtype", None)
    return f"{space.__class__.__name__}(shape={shape}, dtype={dtype})"


def summarize_observation(observation: Any) -> str:
    # observation 可能是 ndarray，也可能是 dict；这里统一压缩成可读摘要。
    shape = getattr(observation, "shape", None)
    dtype = getattr(observation, "dtype", None)
    if shape is not None:
        return f"shape={shape}, dtype={dtype}"
    if isinstance(observation, dict):
        keys = ", ".join(str(key) for key in observation.keys())
        return f"dict(keys=[{keys}])"
    return type(observation).__name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test Farama MaMuJoCo Humanoid through the PettingZoo Parallel API."
    )
    parser.add_argument("--domain", default="Humanoid", help="MaMuJoCo domain name.")
    parser.add_argument(
        "--task",
        default=None,
        help="Deprecated placeholder. MaMuJoCo parallel_env uses domain and partitioning only.",
    )
    parser.add_argument(
        "--partitioning",
        default="none",
        help='Use "none" for single-agent Humanoid or values like "9|8".',
    )
    parser.add_argument("--steps", type=int, default=5, help="Random rollout steps.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument(
        "--render-mode",
        default=None,
        choices=[None, "human", "rgb_array"],
        help="Keep empty for headless smoke tests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    partitioning = parse_partitioning(args.partitioning)
    if args.task is not None:
        raise SystemExit(
            "MaMuJoCo parallel_env does not use a separate task argument. "
            "Run without --task, for example: "
            "python src/check_mamujoco_env.py --partitioning none --steps 5"
        )

    try:
        from gymnasium_robotics import mamujoco_v1
    except Exception as exc:  # pragma: no cover - this is a setup diagnostic.
        raise SystemExit(
            "Failed to import gymnasium_robotics.mamujoco_v1. "
            "Install dependencies with: pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc

    env = mamujoco_v1.parallel_env(args.domain, partitioning, render_mode=args.render_mode)

    try:
        observations, infos = env.reset(seed=args.seed)
        # reset 后先打印 agents 和空间信息，确认 API 形态符合 PPO 代码假设。
        print(f"domain={args.domain}, task={args.task}, partitioning={partitioning}")
        print(f"possible_agents={list(env.possible_agents)}")
        print(f"active_agents={list(env.agents)}")

        for index, agent in enumerate(env.agents):
            env.action_space(agent).seed(args.seed + index)
            print(f"[{agent}] observation_space={summarize_space(env.observation_space(agent))}")
            print(f"[{agent}] action_space={summarize_space(env.action_space(agent))}")
            print(f"[{agent}] first_observation={summarize_observation(observations[agent])}")

        total_rewards = {agent: 0.0 for agent in env.possible_agents}

        for step in range(1, args.steps + 1):
            # 随机动作只用于验证 step 链路，不代表策略性能。
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent, reward in rewards.items():
                total_rewards[agent] = total_rewards.get(agent, 0.0) + float(reward)

            print(
                f"step={step} "
                f"active_agents={list(env.agents)} "
                f"rewards={{{', '.join(f'{agent}: {float(reward):.3f}' for agent, reward in rewards.items())}}}"
            )

            if not env.agents:
                observations, infos = env.reset(seed=args.seed + step)
                print("environment reset after all agents finished")

        print(f"total_random_rollout_rewards={total_rewards}")
        print("smoke_test=passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
