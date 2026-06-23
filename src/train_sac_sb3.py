from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from envs import make_humanoid_gymnasium_env


def default_run_root() -> str:
    # AutoDL 上默认把运行产物放到数据盘；本地调试时退回 runs/。
    if Path("/root/autodl-tmp").exists():
        return "/root/autodl-tmp/Humanoid-runs"
    return "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an SB3 SAC baseline on MaMuJoCo Humanoid.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--target-entropy", default="auto")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument(
        "--net-arch",
        default="256,256",
        help="Comma-separated hidden layer sizes for the SAC actor and critic.",
    )
    parser.add_argument(
        "--normalize-observations",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use SB3 VecNormalize for observations. Use --no-normalize-observations to disable.",
    )
    parser.add_argument("--run-root", default=default_run_root())
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tensorboard-log-dir", default=None)
    parser.add_argument("--tensorboard-run-name", default="sac")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--progress-bar",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable SB3 progress bar if rich/tqdm support is installed.",
    )
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Save replay buffer under checkpoints/. This can be large and should not be committed.",
    )
    return parser


def import_sb3() -> dict[str, Any]:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3/gymnasium dependencies are missing. "
            "On AutoDL, run: pip install -r requirements.txt"
        ) from exc

    return {
        "SAC": SAC,
        "evaluate_policy": evaluate_policy,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "VecNormalize": VecNormalize,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_net_arch(raw_net_arch: str) -> list[int]:
    sizes = [item.strip() for item in raw_net_arch.split(",") if item.strip()]
    if not sizes:
        raise ValueError("--net-arch must contain at least one hidden layer size.")
    return [int(size) for size in sizes]


def create_run_dir(run_root: str, run_name: str | None, seed: int) -> Path:
    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"sac_sb3_humanoid_seed{seed}_{timestamp}"

    run_dir = Path(run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_vec_env(
    *,
    modules: dict[str, Any],
    seed: int,
    monitor_path: Path | None,
    normalize_observations: bool,
):
    Monitor = modules["Monitor"]
    DummyVecEnv = modules["DummyVecEnv"]
    VecNormalize = modules["VecNormalize"]

    def make_env():
        env = make_humanoid_gymnasium_env(seed=seed)
        monitor_file = str(monitor_path) if monitor_path is not None else None
        return Monitor(env, filename=monitor_file)

    vec_env = DummyVecEnv([make_env])
    if normalize_observations:
        # Stage 3 先只归一化 observation，reward 保持原始尺度，方便和 PPO return 对比。
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    return vec_env


def build_eval_env(
    *,
    modules: dict[str, Any],
    seed: int,
    normalize_observations: bool,
    vecnormalize_path: Path,
):
    DummyVecEnv = modules["DummyVecEnv"]
    Monitor = modules["Monitor"]
    VecNormalize = modules["VecNormalize"]

    def make_env():
        return Monitor(make_humanoid_gymnasium_env(seed=seed), filename=None)

    eval_env = DummyVecEnv([make_env])
    if normalize_observations:
        eval_env = VecNormalize.load(str(vecnormalize_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    return eval_env


def write_config(
    *,
    path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    tensorboard_dir: Path,
    observation_dim: int,
    action_dim: int,
    net_arch: list[int],
) -> None:
    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["tensorboard_dir"] = str(tensorboard_dir)
    config["observation_dim"] = observation_dim
    config["action_dim"] = action_dim
    config["net_arch"] = net_arch

    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def write_eval_outputs(
    *,
    run_dir: Path,
    episode_rewards: list[float],
    episode_lengths: list[int],
) -> None:
    mean_return = float(np.mean(episode_rewards))
    std_return = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))

    result = {
        "episodes": len(episode_rewards),
        "returns": [float(value) for value in episode_rewards],
        "lengths": [int(value) for value in episode_lengths],
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
    }

    with (run_dir / "eval_results.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    with (run_dir / "eval_output.txt").open("w", encoding="utf-8") as file:
        for index, (episode_return, episode_length) in enumerate(
            zip(episode_rewards, episode_lengths),
            start=1,
        ):
            file.write(
                f"episode={index} return={episode_return:.3f} length={episode_length}\n"
            )
        file.write(f"mean_return={mean_return:.3f} std_return={std_return:.3f}\n")
        file.write(f"mean_length={mean_length:.3f}\n")

    print(
        f"evaluation_done=true mean_return={mean_return:.3f} "
        f"std_return={std_return:.3f} mean_length={mean_length:.3f}"
    )


def main() -> None:
    args = build_parser().parse_args()
    modules = import_sb3()
    set_seed(args.seed)

    run_dir = create_run_dir(args.run_root, args.run_name, args.seed)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.tensorboard_log_dir) if args.tensorboard_log_dir else run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    net_arch = parse_net_arch(args.net_arch)
    train_env = build_vec_env(
        modules=modules,
        seed=args.seed,
        monitor_path=run_dir / "monitor",
        normalize_observations=args.normalize_observations,
    )
    vecnormalize_path = run_dir / "vecnormalize.pkl"

    try:
        observation_dim = int(np.prod(train_env.observation_space.shape))
        action_dim = int(np.prod(train_env.action_space.shape))
        write_config(
            path=run_dir / "config.json",
            args=args,
            run_dir=run_dir,
            tensorboard_dir=tensorboard_dir,
            observation_dim=observation_dim,
            action_dim=action_dim,
            net_arch=net_arch,
        )

        SAC = modules["SAC"]
        model = SAC(
            policy=args.policy,
            env=train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            target_entropy=args.target_entropy,
            policy_kwargs={"net_arch": net_arch},
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            log_interval=args.log_interval,
            tb_log_name=args.tensorboard_run_name,
            progress_bar=args.progress_bar,
        )

        model_path = checkpoint_dir / "sac_final"
        model.save(str(model_path))
        if args.save_replay_buffer:
            model.save_replay_buffer(str(checkpoint_dir / "replay_buffer.pkl"))
        if args.normalize_observations:
            train_env.save(str(vecnormalize_path))

        eval_env = build_eval_env(
            modules=modules,
            seed=args.eval_seed,
            normalize_observations=args.normalize_observations,
            vecnormalize_path=vecnormalize_path,
        )
        try:
            evaluate_policy = modules["evaluate_policy"]
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )
            write_eval_outputs(
                run_dir=run_dir,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
            )
        finally:
            eval_env.close()

        print(f"training_done=true run_dir={run_dir}")
    finally:
        train_env.close()


if __name__ == "__main__":
    main()
