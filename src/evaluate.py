from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from envs import make_humanoid_single_agent_env
from normalization import RunningMeanStd
from ppo import ActorCritic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def choose_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def load_agent_checkpoint(
    checkpoint_path: Path,
    agent: ActorCritic,
    observation_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    device: torch.device,
) -> RunningMeanStd | None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 兼容旧 checkpoint：旧版本直接保存 agent.state_dict()。
    if "agent_state_dict" not in checkpoint:
        agent.load_state_dict(checkpoint)
        return None

    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.configure_action_squash(
        squash_actions=bool(checkpoint.get("squash_actions", False)),
        action_low=np.asarray(checkpoint.get("action_low", action_low), dtype=np.float32),
        action_high=np.asarray(checkpoint.get("action_high", action_high), dtype=np.float32),
    )

    if not checkpoint.get("normalize_observations", False):
        return None

    obs_rms = RunningMeanStd(shape=(observation_dim,))
    obs_rms.load_state_dict(checkpoint["obs_rms"])
    return obs_rms


def prepare_observation(
    observation: np.ndarray,
    obs_rms: RunningMeanStd | None,
) -> np.ndarray:
    if obs_rms is None:
        return observation.astype(np.float32)

    # 评估阶段只使用训练时保存的统计量，不更新 mean/std。
    return obs_rms.normalize(observation)


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)

    # 评估也用同一个单智能体适配器，避免训练/评估环境不一致。
    env = make_humanoid_single_agent_env(seed=args.seed)
    observation = env.reset()

    # checkpoint 只保存网络参数，网络结构仍由环境维度和 hidden_size 创建。
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = ActorCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_size,
    ).to(device)
    obs_rms = load_agent_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        agent=agent,
        observation_dim=observation_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        device=device,
    )
    agent.eval()

    returns: list[float] = []

    try:
        for episode in range(1, args.episodes + 1):
            observation = env.reset()
            episode_return = 0.0
            episode_length = 0
            done = False

            while not done:
                model_observation = prepare_observation(
                    observation=observation,
                    obs_rms=obs_rms,
                )
                observation_tensor = torch.as_tensor(
                    model_observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    # 评估阶段不用随机采样，直接用确定性动作。
                    action_tensor = agent.get_deterministic_action(observation_tensor)

                action = action_tensor.squeeze(0).cpu().numpy()
                step_result = env.step(action)

                observation = step_result.observation
                episode_return += step_result.reward
                episode_length += 1
                done = step_result.done

            returns.append(episode_return)
            print(
                f"episode={episode} "
                f"return={episode_return:.3f} "
                f"length={episode_length}"
            )

        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        print(f"mean_return={mean_return:.3f} std_return={std_return:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
