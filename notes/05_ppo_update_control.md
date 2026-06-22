# 05 已完成：PPO Update Control

## 本节目标

在保留 observation normalization 的基础上，控制 PPO 每次 update 的策略变化幅度。

上一节 v1 已经证明 obs norm 有效，但也暴露出 update 过猛：

- `approx_kl` 多次达到 `0.08` 到 `0.12`
- `clip_fraction` 经常在 `0.45` 到 `0.57`

因此本节加入 KL early stopping。

## 已完成代码

1. `src/ppo.py`
   - `PPOConfig` 增加 `target_kl`
   - PPO update 每个 epoch 后检查 approximate KL
   - 超过阈值时提前停止后续 epoch
2. `src/train_ppo.py`
   - 新增命令行参数 `--target-kl`
   - 训练日志新增 `update_epochs_used`
   - 训练日志新增 `early_stopped`

## 实验记录

- `experiment_records/ppo_baseline_v2_obsnorm_kl_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `100000`
- normalize observations: `true`
- target KL: `0.03`
- rollout steps: `2048`
- batch size: `256`
- update epochs: `10`
- learning rate: `3e-4`

## 评估结果

```text
episode=1 return=251.392 length=48
episode=2 return=227.405 length=44
episode=3 return=246.576 length=48
episode=4 return=251.100 length=48
episode=5 return=248.454 length=48
mean_return=244.985 std_return=8.967
```

## 本节分析

- KL early stopping 生效，`approx_kl` 从 v1 的 `0.08-0.12` 降到约 `0.03-0.05`。
- `clip_fraction` 从 v1 的 `0.45-0.57` 降到约 `0.25-0.33`。
- 但 evaluation mean return 从 v1 的 `276.612` 降到 `244.985`。
- tail 中 early stopping 几乎每次都触发，`update_epochs_used` 常只有 2-4。
- `target_kl=0.03` 太保守，限制了 actor 和 critic 的学习。

## 本节结论

- KL early stopping 机制正确。
- `target_kl=0.03` 对当前 Humanoid PPO baseline 太紧。
- 下一步应调大 target KL，而不是丢掉 update control。

## 下一节

进入：

- `notes/06_ppo_kl_target_tuning.md`

下一节目标：

1. 保留 observation normalization。
2. 将 target KL 调到 `0.06`。
3. 测试是否能在控制 KL 的同时恢复 v1 的表现。
