from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from envs import make_humanoid_gymnasium_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render deterministic evaluation videos for an SB3 SAC checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path without .zip or with .zip.")
    parser.add_argument(
        "--vecnormalize",
        default=None,
        help="Path to vecnormalize.pkl if observation normalization was used.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for mp4 files. Defaults to <checkpoint run dir>/videos.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser


def import_sb3() -> dict[str, Any]:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3/gymnasium dependencies are missing. "
            "On AutoDL, run: pip install -r requirements.txt"
        ) from exc

    return {
        "SAC": SAC,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "VecNormalize": VecNormalize,
    }


def default_output_dir(checkpoint_path: Path) -> Path:
    run_dir = checkpoint_path.parent.parent
    return run_dir / "videos"


def build_render_env(
    *,
    modules: dict[str, Any],
    seed: int,
    vecnormalize_path: str | None,
):
    Monitor = modules["Monitor"]
    DummyVecEnv = modules["DummyVecEnv"]
    VecNormalize = modules["VecNormalize"]

    def make_env():
        env = make_humanoid_gymnasium_env(seed=seed, render_mode="rgb_array")
        return Monitor(env, filename=None)

    render_env = DummyVecEnv([make_env])
    if vecnormalize_path is not None:
        render_env = VecNormalize.load(vecnormalize_path, render_env)
        render_env.training = False
        render_env.norm_reward = False
    return render_env


def render_frame(vec_env) -> np.ndarray:
    images = vec_env.get_images()
    frame = images[0] if images else None
    if frame is None:
        raise RuntimeError(
            "Environment returned no frame. Make sure render_mode='rgb_array' is available."
        )
    return np.asarray(frame)


def main() -> None:
    args = build_parser().parse_args()
    modules = import_sb3()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_env = build_render_env(
        modules=modules,
        seed=args.seed,
        vecnormalize_path=args.vecnormalize,
    )

    try:
        SAC = modules["SAC"]
        model = SAC.load(str(checkpoint_path), env=render_env, device=args.device)

        for episode in range(1, args.episodes + 1):
            observation = render_env.reset()
            episode_return = 0.0
            episode_length = 0
            done = False

            video_path = output_dir / f"episode_{episode:03d}.mp4"
            writer = imageio.get_writer(video_path, fps=args.fps)

            try:
                writer.append_data(render_frame(render_env))

                while not done and episode_length < args.max_steps:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, dones, _infos = render_env.step(action)
                    episode_return += float(reward[0])
                    episode_length += 1
                    done = bool(dones[0])

                    writer.append_data(render_frame(render_env))
            finally:
                writer.close()

            print(
                f"episode={episode} "
                f"return={episode_return:.3f} "
                f"length={episode_length} "
                f"video={video_path}"
            )
    finally:
        render_env.close()


if __name__ == "__main__":
    main()
