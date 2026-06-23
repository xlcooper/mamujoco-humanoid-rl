# 04 已完成：PPO 诊断与 Observation Normalization

## 本节目标

基于 `ppo_baseline_v0_seed0` 的真实结果，增强 PPO baseline 的稳定性和可分析性。

本节重点是 observation normalization 和更可分析的训练日志。

## 已完成代码

1. `src/normalization.py`
   - 新增 `RunningMeanStd`
   - 在线跟踪 observation 均值和方差
   - 对 observation 做标准化和 clip
2. `src/train_ppo.py`
   - 新增命令行参数 `--normalize-observations`
   - 训练时更新 observation 统计量
   - checkpoint 新增保存 observation normalization 统计量
   - 训练日志新增 `rolling_episode_return`
   - 训练日志新增 `rolling_episode_length`
3. `src/evaluate.py`
   - 加载新 checkpoint 格式
   - 评估时使用训练阶段保存的 observation 统计量
   - 保持旧 checkpoint 兼容

## 实验记录

- `experiment_records/ppo_baseline_v1_obsnorm_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `100000`
- normalize observations: `true`
- rollout steps: `2048`
- batch size: `256`
- update epochs: `10`
- learning rate: `3e-4`

## 评估结果

```text
episode=1 return=279.782 length=54
episode=2 return=265.199 length=50
episode=3 return=280.839 length=54
episode=4 return=281.264 length=54
episode=5 return=275.974 length=53
mean_return=276.612 std_return=6.004
```

## 本节分析

- v1 evaluation mean return 为 `276.612`，高于 v0 的 `243.977`。
- v1 episode length 为 50-54，高于 v0 的 44-47。
- v1 value loss 明显低于 v0，说明 observation normalization 对 critic 有帮助。
- v1 entropy 没有塌缩。
- 但 v1 的 `approx_kl` 和 `clip_fraction` 明显过高，说明策略更新太猛。

## 本节结论

- Observation normalization 有效，应保留。
- 当前主要问题变成 PPO update 幅度过大。
- 下一步应加入 update control，而不是直接加长训练。

## 下一节

进入：

- `notes/05_ppo_update_control.md`

下一节目标：

1. 增加 KL early stopping。
2. 控制 PPO 每轮 update 的策略变化幅度。
3. 跑 v2，与 v1 对比。
