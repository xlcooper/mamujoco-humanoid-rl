# 09 已完成：PPO Relaxed Action Std Control

## 本节目标

继续验证 action std control，把上限从 `0.5` 放宽到 `1.0`。

08 说明 `log_std_max=0.5` 能压低 entropy 和 KL，但会把策略压到固定 18 步倒地。因此本节测试更宽松的 `action_log_std_max=1.0`。

## 已完成实验

- `experiment_records/ppo_long_obsnorm_logstd10_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `3000000`
- normalize observations: `true`
- target KL: 不启用
- action log std min: `-5.0`
- action log std max: `1.0`

## 评估结果

```text
episode=1 return=71.529 length=19
episode=2 return=86.694 length=24
episode=3 return=72.421 length=19
episode=4 return=71.844 length=20
episode=5 return=78.150 length=19
episode=6 return=67.844 length=20
episode=7 return=73.175 length=19
episode=8 return=68.725 length=19
episode=9 return=66.995 length=19
episode=10 return=66.299 length=19
mean_return=72.367 std_return=5.829
```

## 本节分析

和 07、08 对比：

| 指标 | 07 no clamp | 08 max 0.5 | 09 max 1.0 |
| --- | ---: | ---: | ---: |
| evaluation mean return | 326.992 | 78.767 | 72.367 |
| evaluation std | 80.387 | 0.132 | 5.829 |
| evaluation episode length | 50-113 | 18 | 19-24 |
| tail rolling episode return mean | 324.74 | 78.53 | 185.69 |
| tail entropy mean | 54.67 | 20.52 | 39.15 |
| tail approx KL mean | 0.2990 | 0.0535 | 4.2092 |
| tail clip fraction mean | 0.5571 | 0.4520 | 0.7680 |
| tail action log std mean | 未记录 | -0.2121 | 0.8843 |
| tail action log std max | 未记录 | 0.3545 | 1.0000 |

观察：

- `log_std_max=1.0` 没有恢复 07 的表现。
- 09 的 entropy 介于 07 和 08 之间，但 return 仍然崩塌。
- `approx_kl` 和 `clip_fraction` 反而比 07 更糟。
- `action_log_std_max` 长期触顶，说明策略持续想增加探索噪声。

## 本节结论

- 硬性 action log std clamp 路线暂时失败。
- `0.5` 太紧，`1.0` 又导致 KL/clip fraction 爆炸，二者都不是好 baseline。
- 下一步应诊断 raw Gaussian action 被环境裁剪的比例，而不是继续猜 std 上限。

## 下一节

进入：

- `notes/10_ppo_action_clipping_diagnostics.md`

下一节目标：

1. 增加动作裁剪比例日志。
2. 用当前最强配置重跑 `3M` 长训。
3. 判断是否需要实现 tanh-squashed Gaussian policy。
