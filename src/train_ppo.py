from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from envs import make_humanoid_single_agent_env
from ppo import ActorCritic, PPOConfig, RolloutBuffer, update_ppo


def default_run_root() -> str:
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
    parser.add_argument("--run-root", default=default_run_root())
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def choose_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def set_seed(seed: int) -> None:
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
                "mean_reward",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
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
                row["mean_reward"],
                row["policy_loss"],
                row["value_loss"],
                row["entropy"],
                row["approx_kl"],
                row["clip_fraction"],
            ]
        )


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    run_dir = create_run_dir(args.run_root, args.run_name, args.seed)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_header_if_needed(metrics_path)

    env = make_humanoid_single_agent_env(seed=args.seed)
    observation = env.reset()

    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

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
    )

    run_config = vars(args).copy()
    run_config["observation_dim"] = observation_dim
    run_config["action_dim"] = action_dim
    run_config["device"] = str(device)
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

    total_updates = args.total_timesteps // args.rollout_steps
    if total_updates < 1:
        raise ValueError("--total-timesteps must be at least --rollout-steps.")

    try:
        for update in range(1, total_updates + 1):
            buffer = RolloutBuffer(
                rollout_steps=args.rollout_steps,
                observation_dim=config.observation_dim,
                action_dim=config.action_dim,
                device=device,
            )

            rollout_rewards: list[float] = []
            last_done = False

            for _ in range(args.rollout_steps):
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    # Rollout phase: sample action and store log_prob/value from old policy.
                    action_tensor, log_prob_tensor, _, value_tensor = agent.get_action_and_value(
                        observation_tensor
                    )

                action = action_tensor.squeeze(0).cpu().numpy()
                log_prob = float(log_prob_tensor.item())
                value = float(value_tensor.item())

                step_result = env.step(action)

                buffer.add(
                    observation=observation,
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
                    last_episode_return = episode_return
                    last_episode_length = episode_length
                    observation = env.reset()
                    episode_return = 0.0
                    episode_length = 0
                else:
                    observation = step_result.observation

            if last_done:
                last_value = 0.0
            else:
                observation_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    _, _, _, value_tensor = agent.get_action_and_value(observation_tensor)
                last_value = float(value_tensor.item())

            # GAE converts rollout rewards and values into PPO training targets.
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

            mean_reward = float(np.mean(rollout_rewards))
            row = {
                "global_step": global_step,
                "update": update,
                "episode_return": last_episode_return,
                "episode_length": last_episode_length,
                "mean_reward": mean_reward,
                **update_metrics,
            }
            append_metrics(metrics_path, row)

            print(
                "update={update} global_step={global_step} "
                "last_ep_return={episode_return:.3f} "
                "last_ep_len={episode_length} "
                "mean_reward={mean_reward:.3f} "
                "policy_loss={policy_loss:.4f} "
                "value_loss={value_loss:.4f} "
                "entropy={entropy:.4f} "
                "approx_kl={approx_kl:.6f}".format(**row)
            )

            should_save = args.save_every_updates > 0 and update % args.save_every_updates == 0
            if should_save:
                checkpoint_path = checkpoint_dir / f"agent_update_{update}.pt"
                torch.save(agent.state_dict(), checkpoint_path)

        final_checkpoint = checkpoint_dir / "agent_final.pt"
        torch.save(agent.state_dict(), final_checkpoint)
        print(f"training_done=true run_dir={run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
