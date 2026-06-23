# SAC SB3 Smoke Seed0

## 目的

记录 `sac_sb3_smoke_seed0` 的轻量训练结果，用于确认 SB3 SAC smoke test 链路是否跑通。

本记录只提交摘要，不提交 checkpoint、完整 run 目录、replay buffer、TensorBoard events 或视频。

## Run 信息

- run directory: `/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0`
- environment baseline: `AUTODL_HOST_BASELINE.md`
- monitor csv: `/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/monitor.monitor.csv`

## Config

```json
{
  "seed": 0,
  "total_timesteps": 5000,
  "learning_rate": 0.0003,
  "buffer_size": 100000,
  "learning_starts": 500,
  "batch_size": 256,
  "tau": 0.005,
  "gamma": 0.99,
  "train_freq": 1,
  "gradient_steps": 1,
  "ent_coef": "auto",
  "target_entropy": "auto",
  "policy": "MlpPolicy",
  "net_arch": [
    256,
    256
  ],
  "normalize_observations": true,
  "run_root": "/root/autodl-tmp/Humanoid-runs",
  "run_name": "sac_sb3_smoke_seed0",
  "tensorboard_log_dir": null,
  "tensorboard_run_name": "sac",
  "eval_episodes": 3,
  "eval_seed": 10000,
  "log_interval": 4,
  "device": "auto",
  "verbose": 1,
  "progress_bar": false,
  "save_replay_buffer": false,
  "run_dir": "/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0",
  "tensorboard_dir": "/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/tensorboard",
  "observation_dim": 348,
  "action_dim": 17
}
```

## Monitor Tail

保留 SB3 Monitor 最后 20 行：

| r | l | t |
| --- | --- | --- |
| 402.62107 | 86 | 37.224314 |
| 123.586168 | 32 | 37.561271 |
| 176.582738 | 37 | 37.944444 |
| 323.081597 | 78 | 38.77876 |
| 99.821497 | 23 | 39.021743 |
| 436.296124 | 88 | 39.971568 |
| 550.733998 | 114 | 41.177862 |
| 196.988377 | 40 | 41.594678 |
| 436.757876 | 101 | 42.671583 |
| 180.858631 | 35 | 43.043979 |
| 295.307167 | 65 | 43.744575 |
| 151.670864 | 30 | 44.004865 |
| 225.429722 | 42 | 44.42945 |
| 434.963651 | 88 | 45.351287 |
| 297.954498 | 57 | 45.96834 |
| 263.560334 | 54 | 46.557172 |
| 229.430013 | 45 | 47.045367 |
| 226.993338 | 46 | 47.554656 |
| 289.976288 | 57 | 48.183006 |
| 316.744262 | 60 | 48.85887 |


## Evaluation Output

```text
episode=1 return=198.458 length=41
episode=2 return=217.913 length=45
episode=3 return=204.797 length=42
mean_return=207.056 std_return=8.102
mean_length=42.667
```

## Evaluation JSON

```json
{
  "episodes": 3,
  "returns": [
    198.45807886123657,
    217.9133026599884,
    204.79669293761253
  ],
  "lengths": [
    41,
    45,
    42
  ],
  "mean_return": 207.0560248196125,
  "std_return": 8.10164052027514,
  "mean_length": 42.666666666666664
}
```

## 初步观察

- SB3 SAC 训练入口已跑通，`5000` timesteps 内完成 `136` 个 episode，训练日志中出现 rollout 与 train 指标。
- `Monitor` 记录正常，tail episode return 范围约 `99.821` 到 `550.734`，episode length 范围约 `23` 到 `114`。
- deterministic evaluation 可加载 `checkpoints/sac_final.zip` 和 `vecnormalize.pkl`，3 episodes mean return 为 `207.056`，mean length 为 `42.667`。
- 本次 smoke test 只说明工程链路可用，不作为 SAC 性能结论。下一步可以进入更长的 SAC baseline 训练。
