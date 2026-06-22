from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from envs import make_humanoid_single_agent_env
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


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)

    env = make_humanoid_single_agent_env(seed=args.seed)
    observation = env.reset()

    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = ActorCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_size,
    ).to(device)
    agent.load_state_dict(torch.load(Path(args.checkpoint), map_location=device))
    agent.eval()

    returns: list[float] = []

    try:
        for episode in range(1, args.episodes + 1):
            observation = env.reset()
            episode_return = 0.0
            episode_length = 0
            done = False

            while not done:
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    # Evaluation uses the Gaussian mean as a deterministic action.
                    action_mean, _, _ = agent.forward(observation_tensor)

                action = action_mean.squeeze(0).cpu().numpy()
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
