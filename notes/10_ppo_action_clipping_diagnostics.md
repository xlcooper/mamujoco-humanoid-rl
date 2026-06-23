# 10 已完成：PPO Action Clipping Diagnostics

## 本节目标

诊断连续动作高斯策略采样出来的 raw action 有多少被环境动作边界裁剪。

07 的最强长训存在 entropy、KL、clip fraction 过高的问题；08 和 09 说明硬性限制 action log std 会让策略退化。因此本节验证一个更底层的怀疑：

> raw Gaussian action 经常超出环境动作范围，`envs.py` 会把它裁剪后送入 MuJoCo，但 PPO 计算 log_prob 时仍使用未裁剪的 raw action。

## 已完成代码

1. `src/train_ppo.py`
   - rollout 采样时读取环境动作上下界 `env.action_space.low/high`
   - 统计 raw action 超出环境动作边界的维度比例
   - 统计 raw action 超出边界的平均幅度
   - 训练日志新增 `action_clip_fraction`
   - 训练日志新增 `action_clip_excess_mean`
   - 终端输出新增 `act_clip`

这两个指标只用于诊断，不改变训练行为。

## 已完成实验

- `experiment_records/ppo_long_obsnorm_clipdiag_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `3000000`
- normalize observations: `true`
- target KL: 不启用
- action log std clamp: 不启用

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

和 07 长训基线对比：

| 指标 | 07 long obs norm | 10 clipdiag |
| --- | ---: | ---: |
| evaluation mean return | 326.992 | 326.992 |
| evaluation std | 80.387 | 80.387 |
| tail entropy mean | 54.67 | 54.67 |
| tail approx KL mean | 0.2990 | 0.2990 |
| tail PPO clip fraction mean | 0.5571 | 0.5571 |
| tail action clip fraction mean | 未记录 | 0.9826 |
| tail action clip excess mean | 未记录 | 19.55 |
| tail action log std mean | 未记录 | 1.7976 |
| tail action log std max | 未记录 | 2.4999 |

观察：

- 本实验只加诊断，主要训练和评估指标与 07 完全一致，说明诊断代码没有改变训练行为。
- `action_clip_fraction` 约 `0.9826`，几乎所有 raw action 维度都被环境裁剪。
- `action_clip_excess_mean` 约 `19.55`，越界幅度极大。
- 平均 `log_std` 约 `1.7976`，对应 std 约 `6.04`；最大 `log_std` 约 `2.4999`，对应 std 约 `12.18`。

## 本节结论

- 当前无界 Gaussian policy 和环境动作边界存在严重不匹配。
- PPO 计算 log_prob 的 raw action 与环境实际执行的 clipped action 大量不一致。
- 下一步应实现 tanh-squashed Gaussian policy，让策略天然输出合法动作。

## 下一节

进入：

- `notes/11_ppo_tanh_squashed_policy.md`

下一节目标：

1. 增加可选 `--squash-actions`。
2. 让 actor 先采样 raw Gaussian action，再用 tanh 映射到环境动作范围。
3. PPO log_prob 使用 tanh 变换的 Jacobian 修正。
4. 跑 `3M` timesteps 对比实验。
