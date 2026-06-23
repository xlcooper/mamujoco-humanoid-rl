# 12 已完成：PPO Squashed Update Epochs Tuning

## 本节目标

在 tanh-squashed Gaussian policy 上降低 PPO update 强度。

11 已经解决动作越界问题：

- `action_clip_fraction`: `0.9826 -> 0.0000`
- `action_clip_excess_mean`: `19.55 -> 0.0000`

但 11 也暴露了新问题：

- evaluation mean return 从 `326.992` 降到 `283.664`
- tail `approx_kl` 从 `0.2990` 升到 `1.1188`
- tail PPO `clip_fraction` 从 `0.5571` 升到 `0.8774`

因此本节保留 `--squash-actions`，将 `update_epochs` 从 `10` 降到 `4`。

## 已完成实验

- `experiment_records/ppo_long_obsnorm_squash_ep4_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `3000000`
- normalize observations: `true`
- squash actions: `true`
- update epochs: `4`
- target KL: 不启用
- action log std clamp: 不启用

## 评估结果

```text
episode=1 return=856.341 length=160
episode=2 return=583.489 length=113
episode=3 return=655.889 length=133
episode=4 return=761.233 length=145
episode=5 return=635.018 length=126
episode=6 return=989.033 length=193
episode=7 return=679.054 length=139
episode=8 return=688.288 length=144
episode=9 return=656.312 length=132
episode=10 return=655.458 length=131
mean_return=716.011 std_return=115.490
```

## 本节分析

和 07、11 对比：

| 指标 | 07 no squash ep10 | 11 squash ep10 | 12 squash ep4 |
| --- | ---: | ---: | ---: |
| evaluation mean return | 326.992 | 283.664 | 716.011 |
| evaluation std | 80.387 | 13.380 | 115.490 |
| evaluation episode length | 50-113 | 54-60 | 113-193 |
| tail rolling episode return mean | 324.74 | 296.44 | 617.98 |
| tail rolling episode length mean | 68.23 | 59.26 | 121.45 |
| tail value loss mean | 125.95 | 64.08 | 144.24 |
| tail entropy mean | 54.67 | 32.22 | 26.64 |
| tail approx KL mean | 0.2990 | 1.1188 | 0.1038 |
| tail PPO clip fraction mean | 0.5571 | 0.8774 | 0.4198 |
| tail action clip fraction mean | 0.9826 | 0.0000 | 0.0000 |

观察：

- `update_epochs=4` 让 squashed policy 的 KL 和 PPO clip fraction 明显下降。
- evaluation mean return 大幅提升到 `716.011`。
- 动作越界问题继续保持为 `0`。
- episode length 明显变长，策略已经能更稳定地移动更久。
- evaluation std 仍较高，因此需要多 seed 验证。

## 本节结论

- `observation normalization + tanh-squashed Gaussian policy + update_epochs=4` 是当前最强候选 baseline。
- 12 不是单纯提高 return，而是同时解决了动作越界，并把 update 强度降到更合理水平。
- 下一步应做多 seed 验证，而不是继续添加新技巧。

## 下一节

进入：

- `notes/stage1_2_ppo/13_ppo_squashed_ep4_multiseed.md`

下一节目标：

1. 固定 12 的候选配置。
2. 跑 seed `1` 和 seed `2`。
3. 汇总 seed 0/1/2 的 return、episode length、KL、clip fraction 和动作裁剪指标。
4. 判断 Stage 1 是否可以收束。
