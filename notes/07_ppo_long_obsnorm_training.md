# 07 已完成：PPO Long Obsnorm Training

## 本节目标

把目前最强的短训配置放大成一个真正的大实验。

前面 `100k` timesteps 的 v0-v3 主要用于验证代码、诊断问题和选择配置。本节使用当前最强短训配置：

- observation normalization
- no target KL
- seed 0
- `3M` timesteps

## 已完成实验

- `experiment_records/ppo_long_obsnorm_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `3000000`
- normalize observations: `true`
- target KL: 不启用
- rollout steps: `2048`
- batch size: `256`
- update epochs: `10`
- learning rate: `3e-4`

## 评估结果

```text
episode=1 return=240.098 length=53
episode=2 return=322.672 length=74
episode=3 return=325.133 length=72
episode=4 return=292.005 length=66
episode=5 return=328.724 length=70
episode=6 return=230.147 length=50
episode=7 return=261.603 length=57
episode=8 return=356.647 length=78
episode=9 return=394.110 length=92
episode=10 return=518.780 length=113
mean_return=326.992 std_return=80.387
```

## 本节分析

和短训 v1 对比：

| 指标 | v1 obs norm 100k | long obs norm 3M |
| --- | ---: | ---: |
| evaluation mean return | 276.612 | 326.992 |
| evaluation std | 6.004 | 80.387 |
| evaluation episode length | 50-54 | 50-113 |
| tail rolling episode return mean | 307.98 | 324.74 |
| tail value loss mean | 109.38 | 125.95 |
| tail entropy mean | 24.39 | 54.67 |
| tail approx KL mean | 0.0934 | 0.2990 |
| tail clip fraction mean | 0.5028 | 0.5571 |

观察：

- 长训让 evaluation mean return 从 `276.612` 提升到 `326.992`，说明 baseline 有继续学习能力。
- 最好 episode return 达到 `518.780`，episode length 达到 `113`，策略能跑出更长行为。
- 但评估标准差达到 `80.387`，稳定性明显不足。
- entropy 从约 `24` 飙升到 `54+`，说明高斯策略的动作标准差被学得过大。
- `approx_kl` 和 `clip_fraction` 都偏高，策略更新幅度仍然过猛。

## 本节结论

- `observation normalization + long training` 是有效主线。
- 但当前长训不是一个干净稳定的最终 baseline。
- 下一步优先修动作探索噪声过大的问题，而不是马上做多 seed。

## 下一节

进入：

- `notes/08_ppo_action_std_control.md`

下一节目标：

1. 为训练日志增加 action log std 诊断。
2. 增加可选的 action log std clamp。
3. 跑同样 `3M` timesteps 的对比实验，观察是否降低 entropy、KL 和 clip fraction，并提升稳定性。
