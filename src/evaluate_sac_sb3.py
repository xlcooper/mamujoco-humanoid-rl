from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from envs import make_humanoid_gymnasium_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an SB3 SAC checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path without .zip or with .zip.")
    parser.add_argument(
        "--vecnormalize",
        default=None,
        help="Path to vecnormalize.pkl if observation normalization was used.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-json", default=None)
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


def build_eval_env(
    *,
    modules: dict[str, Any],
    seed: int,
    vecnormalize_path: str | None,
):
    Monitor = modules["Monitor"]
    DummyVecEnv = modules["DummyVecEnv"]
    VecNormalize = modules["VecNormalize"]

    def make_env():
        return Monitor(make_humanoid_gymnasium_env(seed=seed), filename=None)

    eval_env = DummyVecEnv([make_env])
    if vecnormalize_path is not None:
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    return eval_env


def save_json(
    *,
    output_path: Path,
    returns: list[float],
    lengths: list[int],
) -> None:
    result = {
        "episodes": len(returns),
        "returns": returns,
        "lengths": lengths,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    modules = import_sb3()

    eval_env = build_eval_env(
        modules=modules,
        seed=args.seed,
        vecnormalize_path=args.vecnormalize,
    )

    try:
        SAC = modules["SAC"]
        model = SAC.load(args.checkpoint, env=eval_env, device=args.device)

        returns: list[float] = []
        lengths: list[int] = []

        for episode in range(1, args.episodes + 1):
            observation = eval_env.reset()
            episode_return = 0.0
            episode_length = 0
            done = False

            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, dones, _infos = eval_env.step(action)
                episode_return += float(reward[0])
                episode_length += 1
                done = bool(dones[0])

            returns.append(episode_return)
            lengths.append(episode_length)
            print(
                f"episode={episode} "
                f"return={episode_return:.3f} "
                f"length={episode_length}"
            )

        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        mean_length = float(np.mean(lengths))
        print(f"mean_return={mean_return:.3f} std_return={std_return:.3f}")
        print(f"mean_length={mean_length:.3f}")

        if args.output_json is not None:
            save_json(
                output_path=Path(args.output_json),
                returns=returns,
                lengths=lengths,
            )
    finally:
        eval_env.close()


if __name__ == "__main__":
    main()
