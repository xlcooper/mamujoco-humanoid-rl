# SAC SB3 1m Seed0

## 目的

记录 `sac_sb3_1m_seed0` 的轻量训练结果，用于确认 SB3 SAC smoke test 链路是否跑通。

本记录只提交摘要，不提交 checkpoint、完整 run 目录、replay buffer、TensorBoard events 或视频。

## Run 信息

- run directory: `/root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0`
- environment baseline: `AUTODL_HOST_BASELINE.md`
- monitor csv: `/root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/monitor.monitor.csv`

## Config

```json
{
  "seed": 0,
  "total_timesteps": 1000000,
  "learning_rate": 0.0003,
  "buffer_size": 1000000,
  "learning_starts": 10000,
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
  "run_name": "sac_sb3_1m_seed0",
  "tensorboard_log_dir": null,
  "tensorboard_run_name": "sac",
  "eval_episodes": 10,
  "eval_seed": 10000,
  "log_interval": 10,
  "device": "auto",
  "verbose": 1,
  "progress_bar": false,
  "save_replay_buffer": false,
  "run_dir": "/root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0",
  "tensorboard_dir": "/root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/tensorboard",
  "observation_dim": 348,
  "action_dim": 17
}
```

## Monitor Tail

保留 SB3 Monitor 最后 20 行：

| r | l | t |
| --- | --- | --- |
| 6074.019232 | 1000 | 10556.093981 |
| 6059.54586 | 1000 | 10567.353812 |
| 5878.486297 | 1000 | 10578.482911 |
| 5879.739585 | 1000 | 10589.455684 |
| 5853.904909 | 1000 | 10600.579112 |
| 6081.52488 | 1000 | 10611.696781 |
| 5959.702263 | 1000 | 10622.881804 |
| 3464.561975 | 581 | 10629.143136 |
| 5850.689062 | 1000 | 10638.181882 |
| 5971.232823 | 1000 | 10647.184933 |
| 5995.823413 | 1000 | 10656.160607 |
| 6052.322386 | 1000 | 10667.294454 |
| 6053.972356 | 1000 | 10678.18271 |
| 5938.418611 | 1000 | 10689.636982 |
| 6024.010538 | 1000 | 10699.359896 |
| 5963.10866 | 1000 | 10710.768082 |
| 5960.317023 | 1000 | 10722.04037 |
| 5788.433872 | 1000 | 10733.645877 |
| 6007.605905 | 1000 | 10745.237532 |
| 5912.462611 | 1000 | 10756.437292 |


## Evaluation Output

```text
episode=1 return=6086.414 length=1000
episode=2 return=6043.676 length=1000
episode=3 return=6016.683 length=1000
episode=4 return=6070.044 length=1000
episode=5 return=6017.678 length=1000
episode=6 return=6075.439 length=1000
episode=7 return=6035.541 length=1000
episode=8 return=6097.653 length=1000
episode=9 return=5977.473 length=1000
episode=10 return=6003.000 length=1000
mean_return=6042.360 std_return=37.329
mean_length=1000.000
```

## Evaluation JSON

```json
{
  "episodes": 10,
  "returns": [
    6086.413942813873,
    6043.676155537367,
    6016.683376967907,
    6070.043651580811,
    6017.6784126758575,
    6075.438585400581,
    6035.540935397148,
    6097.6532661914825,
    5977.472895622253,
    6003.000244438648
  ],
  "lengths": [
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1000
  ],
  "mean_return": 6042.360146662593,
  "std_return": 37.32948009725312,
  "mean_length": 1000.0
}
```

## 初步观察

- TODO: 本地 pull 后确认训练、checkpoint、VecNormalize、TensorBoard 和 evaluation 链路是否完整。
- TODO: 判断 smoke test 是否可以固化为已完成，并决定下一步进入 SAC 长训还是先修参数/环境问题。
