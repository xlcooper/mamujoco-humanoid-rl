from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from envs import SingleAgentMaMuJoCoEnv
from evaluate import choose_device, load_agent_checkpoint, prepare_observation
from ppo import ActorCritic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render deterministic evaluation videos for a trained PPO checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to agent_final.pt.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for mp4 files. Defaults to <checkpoint run dir>/videos.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of videos to record.")
    parser.add_argument("--seed", type=int, default=20_000, help="Evaluation seed.")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Safety limit per episode; Humanoid normally ends earlier.",
    )
    return parser


def default_output_dir(checkpoint_path: Path) -> Path:
    # checkpoint 通常在 <run_dir>/checkpoints/agent_final.pt。
    run_dir = checkpoint_path.parent.parent
    return run_dir / "videos"


def render_frame(env: SingleAgentMaMuJoCoEnv) -> np.ndarray:
    frame = env.env.render()
    if frame is None:
        raise RuntimeError(
            "Environment returned no frame. Make sure render_mode='rgb_array' is available."
        )
    return np.asarray(frame)


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 渲染需要 rgb_array；训练和普通评估默认不打开渲染，避免拖慢速度。
    env = SingleAgentMaMuJoCoEnv(
        domain="Humanoid",
        partitioning=None,
        seed=args.seed,
        render_mode="rgb_array",
    )
    observation = env.reset()

    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = ActorCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_size,
    ).to(device)
    obs_rms = load_agent_checkpoint(
        checkpoint_path=checkpoint_path,
        agent=agent,
        observation_dim=observation_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        device=device,
    )
    agent.eval()

    try:
        for episode in range(1, args.episodes + 1):
            observation = env.reset()
            episode_return = 0.0
            episode_length = 0
            done = False

            video_path = output_dir / f"episode_{episode:03d}.mp4"
            writer = imageio.get_writer(video_path, fps=args.fps)

            try:
                # reset 后先写入第一帧，视频能看到初始姿态。
                writer.append_data(render_frame(env))

                while not done and episode_length < args.max_steps:
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
                        # 展示视频用 deterministic action，避免随机采样让行为忽好忽坏。
                        action_tensor = agent.get_deterministic_action(observation_tensor)

                    action = action_tensor.squeeze(0).cpu().numpy()
                    step_result = env.step(action)

                    observation = step_result.observation
                    episode_return += step_result.reward
                    episode_length += 1
                    done = step_result.done

                    writer.append_data(render_frame(env))
            finally:
                writer.close()

            print(
                f"episode={episode} "
                f"return={episode_return:.3f} "
                f"length={episode_length} "
                f"video={video_path}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
