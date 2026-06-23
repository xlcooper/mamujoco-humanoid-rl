from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from envs import make_humanoid_single_agent_env
from normalization import RunningMeanStd
from ppo import ActorCritic, PPOConfig, RolloutBuffer, update_ppo


def default_run_root() -> str:
    # AutoDL 上默认把运行产物放到数据盘；本地调试时退回 runs/。
    if Path("/root/autodl-tmp").exists():
        return "/root/autodl-tmp/Humanoid-runs"
    return "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a beginner-friendly PPO baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--action-log-std-min", type=float, default=None)
    parser.add_argument("--action-log-std-max", type=float, default=None)
    parser.add_argument("--run-root", default=default_run_root())
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--normalize-observations",
        action="store_true",
        help="Use running mean/std to normalize observations before the PPO network.",
    )
    return parser


def choose_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def set_seed(seed: int) -> None:
    # 固定随机种子，方便复现实验差异。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(run_root: str, run_name: str | None, seed: int) -> Path:
    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_humanoid_seed{seed}_{timestamp}"

    run_dir = Path(run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_header_if_needed(csv_path: Path) -> None:
    # metrics.csv 是 Git 实验记录的原始来源之一，但完整文件留在 AutoDL。
    if csv_path.exists():
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "global_step",
                "update",
                "episode_return",
                "episode_length",
                "rolling_episode_return",
                "rolling_episode_length",
                "mean_reward",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "update_epochs_used",
                "early_stopped",
                "action_log_std_mean",
                "action_log_std_min",
                "action_log_std_max",
            ]
        )


def append_metrics(csv_path: Path, row: dict[str, float | int]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                row["global_step"],
                row["update"],
                row["episode_return"],
                row["episode_length"],
                row["rolling_episode_return"],
                row["rolling_episode_length"],
                row["mean_reward"],
                row["policy_loss"],
                row["value_loss"],
                row["entropy"],
                row["approx_kl"],
                row["clip_fraction"],
                row["update_epochs_used"],
                row["early_stopped"],
                row["action_log_std_mean"],
                row["action_log_std_min"],
                row["action_log_std_max"],
            ]
        )


def prepare_observation(
    observation: np.ndarray,
    obs_rms: RunningMeanStd | None,
    update_stats: bool,
) -> np.ndarray:
    if obs_rms is None:
        return observation.astype(np.float32)

    if update_stats:
        obs_rms.update(observation[None, :])

    return obs_rms.normalize(observation)


def save_checkpoint(
    path: Path,
    agent: ActorCritic,
    normalize_observations: bool,
    obs_rms: RunningMeanStd | None,
) -> None:
    # checkpoint 同时保存网络参数和 observation normalization 状态。
    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "normalize_observations": normalize_observations,
        "obs_rms": obs_rms.state_dict() if obs_rms is not None else None,
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    # 每次训练单独一个 run_dir，里面保存 config、metrics 和 checkpoints。
    run_dir = create_run_dir(args.run_root, args.run_name, args.seed)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_header_if_needed(metrics_path)

    env = make_humanoid_single_agent_env(seed=args.seed)
    observation = env.reset()

    # 从环境空间自动读取维度，避免把 348/17 写死在训练代码里。
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    obs_rms = None
    if args.normalize_observations:
        obs_rms = RunningMeanStd(shape=(observation_dim,))

    config = PPOConfig(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        action_log_std_min=args.action_log_std_min,
        action_log_std_max=args.action_log_std_max,
    )

    run_config = vars(args).copy()
    run_config["observation_dim"] = observation_dim
    run_config["action_dim"] = action_dim
    run_config["device"] = str(device)

    # config.json 记录本次训练设置，后续 experiment_records 会引用它。
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    agent = ActorCritic(
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
        hidden_size=config.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    global_step = 0
    update = 0
    episode_return = 0.0
    episode_length = 0
    last_episode_return = 0.0
    last_episode_length = 0
    recent_episode_returns: deque[float] = deque(maxlen=20)
    recent_episode_lengths: deque[int] = deque(maxlen=20)

    total_updates = args.total_timesteps // args.rollout_steps
    if total_updates < 1:
        raise ValueError("--total-timesteps must be at least --rollout-steps.")

    try:
        # 外层循环：每个 update 先采样一段 rollout，再用这段数据更新策略。
        for update in range(1, total_updates + 1):
            buffer = RolloutBuffer(
                rollout_steps=args.rollout_steps,
                observation_dim=config.observation_dim,
                action_dim=config.action_dim,
                device=device,
            )

            rollout_rewards: list[float] = []
            last_done = False

            # Rollout 阶段：用当前策略和环境交互，收集 PPO 训练所需数据。
            for _ in range(args.rollout_steps):
                model_observation = prepare_observation(
                    observation=observation,
                    obs_rms=obs_rms,
                    update_stats=True,
                )
                observation_tensor = torch.as_tensor(
                    model_observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    # 采样时保存 old log_prob 和 old value，之后 PPO update 会用到。
                    action_tensor, log_prob_tensor, _, value_tensor = agent.get_action_and_value(
                        observation_tensor
                    )

                action = action_tensor.squeeze(0).cpu().numpy()
                log_prob = float(log_prob_tensor.item())
                value = float(value_tensor.item())

                step_result = env.step(action)

                buffer.add(
                    observation=model_observation,
                    action=action,
                    log_prob=log_prob,
                    reward=step_result.reward,
                    done=step_result.done,
                    value=value,
                )

                global_step += 1
                episode_return += step_result.reward
                episode_length += 1
                rollout_rewards.append(step_result.reward)
                last_done = step_result.done

                if step_result.done:
                    # episode 结束后记录最近一局回报，并重置环境进入下一局。
                    last_episode_return = episode_return
                    last_episode_length = episode_length
                    recent_episode_returns.append(episode_return)
                    recent_episode_lengths.append(episode_length)
                    observation = env.reset()
                    episode_return = 0.0
                    episode_length = 0
                else:
                    observation = step_result.observation

            if last_done:
                # rollout 最后一步已经终止，不再 bootstrap value。
                last_value = 0.0
            else:
                # rollout 截断但 episode 未结束，用 critic 估计最后状态价值。
                model_observation = prepare_observation(
                    observation=observation,
                    obs_rms=obs_rms,
                    update_stats=False,
                )
                observation_tensor = torch.as_tensor(
                    model_observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    _, _, _, value_tensor = agent.get_action_and_value(observation_tensor)
                last_value = float(value_tensor.item())

            # GAE 把 rewards + values 转成 advantage 和 return 两类训练目标。
            buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            update_metrics = update_ppo(
                agent=agent,
                optimizer=optimizer,
                buffer=buffer,
                config=config,
                batch_size=args.batch_size,
                update_epochs=args.update_epochs,
            )

            # 训练日志：终端打印最近状态，CSV 保存完整 update 级指标。
            mean_reward = float(np.mean(rollout_rewards))
            rolling_episode_return = 0.0
            rolling_episode_length = 0.0
            if recent_episode_returns:
                rolling_episode_return = float(np.mean(recent_episode_returns))
                rolling_episode_length = float(np.mean(recent_episode_lengths))

            row = {
                "global_step": global_step,
                "update": update,
                "episode_return": last_episode_return,
                "episode_length": last_episode_length,
                "rolling_episode_return": rolling_episode_return,
                "rolling_episode_length": rolling_episode_length,
                "mean_reward": mean_reward,
                **update_metrics,
                **agent.action_log_std_metrics(),
            }
            append_metrics(metrics_path, row)

            print(
                "update={update} global_step={global_step} "
                "last_ep_return={episode_return:.3f} "
                "last_ep_len={episode_length} "
                "roll_return={rolling_episode_return:.3f} "
                "roll_len={rolling_episode_length:.1f} "
                "mean_reward={mean_reward:.3f} "
                "policy_loss={policy_loss:.4f} "
                "value_loss={value_loss:.4f} "
                "entropy={entropy:.4f} "
                "approx_kl={approx_kl:.6f} "
                "epochs_used={update_epochs_used:.0f} "
                "early_stop={early_stopped:.0f} "
                "log_std_mean={action_log_std_mean:.3f}".format(**row)
            )

            should_save = args.save_every_updates > 0 and update % args.save_every_updates == 0
            if should_save:
                # 中间 checkpoint 方便长训练中断后查看，但不提交到 Git。
                checkpoint_path = checkpoint_dir / f"agent_update_{update}.pt"
                save_checkpoint(
                    path=checkpoint_path,
                    agent=agent,
                    normalize_observations=args.normalize_observations,
                    obs_rms=obs_rms,
                )

        final_checkpoint = checkpoint_dir / "agent_final.pt"
        save_checkpoint(
            path=final_checkpoint,
            agent=agent,
            normalize_observations=args.normalize_observations,
            obs_rms=obs_rms,
        )
        print(f"training_done=true run_dir={run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
