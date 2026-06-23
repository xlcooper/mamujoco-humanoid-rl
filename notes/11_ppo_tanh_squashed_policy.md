# 11 已完成：PPO Tanh-Squashed Gaussian Policy

## 本节目标

解决 raw Gaussian action 大量越界的问题。

10 的诊断显示：

- tail `action_clip_fraction` 约 `0.9826`
- tail `action_clip_excess_mean` 约 `19.55`
- tail `action_log_std_mean` 约 `1.7976`
- tail `action_log_std_max` 约 `2.4999`

这说明当前无界 Gaussian policy 采样出的 raw action 几乎总是超出环境动作范围，然后被 `envs.py` 硬裁剪。PPO 训练时计算的是 raw action 的 log_prob，但环境实际执行的是 clipped action，二者不一致。

本节引入 tanh-squashed Gaussian policy，让策略天然输出环境合法动作。

## 已完成代码

1. `src/ppo.py`
   - `ActorCritic` 增加 `squash_actions` 开关（压缩动作）
   - `ActorCritic` 保存环境动作边界对应的 `action_scale` 和 `action_bias`
   - 新增 `squash_raw_action(...)`
   - 新增 `unsquash_action(...)`
   - 新增 `squashed_log_prob(...)`
   - `get_action_and_value(...)` 支持 tanh-squashed action 采样和 log_prob 计算
   - 新增 `get_deterministic_action(...)`，评估时对 mean action 也执行 tanh 映射
2. `src/train_ppo.py`
   - 新增命令行参数 `--squash-actions`
   - 创建 `ActorCritic` 时传入环境动作上下界
   - checkpoint 新增保存 `squash_actions`、`action_low`、`action_high`
3. `src/evaluate.py`
   - 加载 checkpoint 时恢复 `squash_actions` 和动作边界
   - 评估时使用 `agent.get_deterministic_action(...)`

默认不传 `--squash-actions` 时，旧实验行为保持不变。

## 已完成实验

- `experiment_records/ppo_long_obsnorm_squash_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `3000000`
- normalize observations: `true`
- squash actions: `true`
- target KL: 不启用
- action log std clamp: 不启用
- update epochs: `10`

## 评估结果

```text
episode=1 return=278.745 length=56
episode=2 return=306.864 length=60
episode=3 return=275.209 length=55
episode=4 return=283.913 length=57
episode=5 return=300.770 length=59
episode=6 return=288.241 length=57
episode=7 return=295.729 length=58
episode=8 return=275.196 length=55
episode=9 return=266.453 length=54
episode=10 return=265.517 length=54
mean_return=283.664 std_return=13.380
```

## 本节分析

和 07/10 对比：

| 指标 | 07/10 no squash | 11 squash |
| --- | ---: | ---: |
| evaluation mean return | 326.992 | 283.664 |
| evaluation std | 80.387 | 13.380 |
| evaluation episode length | 50-113 | 54-60 |
| tail rolling episode return mean | 324.74 | 296.44 |
| tail value loss mean | 125.95 | 64.08 |
| tail entropy mean | 54.67 | 32.22 |
| tail approx KL mean | 0.2990 | 1.1188 |
| tail PPO clip fraction mean | 0.5571 | 0.8774 |
| tail action clip fraction mean | 0.9826 | 0.0000 |
| tail action clip excess mean | 19.55 | 0.0000 |

观察：

- tanh-squashed policy 完全解决 action clipping 问题。
- 评估稳定性、entropy 和 value loss 都明显改善。
- 但 return 低于 07，且 `approx_kl`、PPO `clip_fraction` 过高。
- 当前 `update_epochs=10` 对 squashed policy 来说过猛。

## 本节结论

- tanh-squashed policy 是正确的结构修复，应继续保留。
- 当前瓶颈从“动作越界”转为“squashed policy 的 PPO update 过强”。
- 下一步应降低 update epochs，而不是回到无界 Gaussian 或继续调 action std clamp。

## 下一节

进入：

- `notes/12_ppo_squashed_update_epochs_tuning.md`

下一节目标：

1. 保留 `--squash-actions`。
2. 将 `update_epochs` 从 `10` 降到 `4`。
3. 跑同样 `3M` timesteps，对比 return、KL、clip fraction 和稳定性。
