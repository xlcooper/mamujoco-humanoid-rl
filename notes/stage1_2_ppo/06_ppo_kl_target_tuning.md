# 06 已完成：PPO KL Target Tuning

## 本节目标

调整 KL early stopping 的阈值，在“更新稳定”和“学习速度”之间找到更好的折中。

上一节 v2 使用 `target_kl=0.03`，虽然降低了 KL 和 clip fraction，但评估表现从 v1 的 `276.612` 回落到 `244.985`。因此本节尝试更宽松的 `target_kl=0.06`。

## 已完成实验

- `experiment_records/ppo_baseline_v3_obsnorm_kl006_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `100000`
- normalize observations: `true`
- target KL: `0.06`
- rollout steps: `2048`
- batch size: `256`
- update epochs: `10`
- learning rate: `3e-4`

## 评估结果

```text
episode=1 return=280.254 length=53
episode=2 return=224.528 length=44
episode=3 return=276.856 length=53
episode=4 return=279.168 length=53
episode=5 return=277.406 length=53
mean_return=267.642 std_return=21.591
```

## 本节分析

和 v1、v2 对比：

| 指标 | v1 obs norm | v2 KL 0.03 | v3 KL 0.06 |
| --- | ---: | ---: | ---: |
| evaluation mean return | 276.612 | 244.985 | 267.642 |
| evaluation std | 6.004 | 8.967 | 21.591 |
| tail approx KL mean | 0.0934 | 0.0350 | 0.0645 |
| tail clip fraction mean | 0.5028 | 0.2863 | 0.4302 |
| tail value loss mean | 109.38 | 552.53 | 156.09 |
| tail update epochs used mean | 10.0 | 3.45 | 5.80 |

观察：

- `target_kl=0.06` 明显好于 `0.03`，评估均值从 `244.985` 回升到 `267.642`。
- KL 和 clip fraction 介于 v1 与 v2 之间，说明 update control 方向有效。
- value loss 恢复到接近 v1 的水平，说明 critic 不再像 v2 那样训练不足。
- 但 v3 仍低于 v1，且评估波动更大。
- tail 中 early stopping 仍全部触发，说明 KL 控制仍会限制训练步幅。

## 本节结论

- `target_kl=0.06` 是比 `0.03` 更合理的阈值。
- 但当前 `100k` 短训下，最强配置仍是 `observation normalization + no target_kl`。
- 暂时不应把 KL early stopping 作为主线长训配置。

## 下一节

进入：

- `notes/stage1_2_ppo/07_ppo_long_obsnorm_training.md`

下一节目标：

1. 使用当前最强短训配置：observation normalization，不启用 target KL。
2. 将训练规模提升到 `3M` timesteps。
3. 得到一条真正适合分析学习曲线、稳定性和最终表现的长训 baseline。
