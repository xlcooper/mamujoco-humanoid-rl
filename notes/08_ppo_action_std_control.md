# 08 已完成：PPO Action Std Control

## 本节目标

控制连续动作高斯策略的探索噪声。

07 的 `3M` 长训证明 baseline 会继续提升，但也暴露出明显问题：

- evaluation mean return: `326.992`
- evaluation std: `80.387`
- tail entropy mean: `54.67`
- tail approx KL mean: `0.2990`
- tail clip fraction mean: `0.5571`

因此本节增加 action log std 诊断，并尝试 `action_log_std_max=0.5`。

## 已完成代码

1. `src/ppo.py`
   - `PPOConfig` 增加 `action_log_std_min`
   - `PPOConfig` 增加 `action_log_std_max`
   - `ActorCritic` 增加 `clamp_action_log_std(...)`
   - `ActorCritic` 增加 `action_log_std_metrics()`
   - PPO update 中 `optimizer.step()` 后执行可选 log std clamp
2. `src/train_ppo.py`
   - 新增命令行参数 `--action-log-std-min`
   - 新增命令行参数 `--action-log-std-max`
   - 训练日志新增 `action_log_std_mean`
   - 训练日志新增 `action_log_std_min`
   - 训练日志新增 `action_log_std_max`
   - 终端输出新增 `log_std_mean`

默认不传 clamp 参数时，旧实验行为保持不变。

## 已完成实验

- `experiment_records/ppo_long_obsnorm_logstd05_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `3000000`
- normalize observations: `true`
- target KL: 不启用
- action log std min: `-5.0`
- action log std max: `0.5`

## 评估结果

```text
episode=1 return=78.515 length=18
episode=2 return=78.708 length=18
episode=3 return=78.628 length=18
episode=4 return=78.952 length=18
episode=5 return=78.810 length=18
episode=6 return=78.774 length=18
episode=7 return=78.810 length=18
episode=8 return=78.789 length=18
episode=9 return=78.704 length=18
episode=10 return=78.983 length=18
mean_return=78.767 std_return=0.132
```

## 本节分析

和 07 长训基线对比：

| 指标 | long obs norm | log std max 0.5 |
| --- | ---: | ---: |
| evaluation mean return | 326.992 | 78.767 |
| evaluation std | 80.387 | 0.132 |
| evaluation episode length | 50-113 | 18 |
| tail rolling episode return mean | 324.74 | 78.53 |
| tail entropy mean | 54.67 | 20.52 |
| tail approx KL mean | 0.2990 | 0.0535 |
| tail clip fraction mean | 0.5571 | 0.4520 |
| tail action log std mean | 未记录 | -0.2121 |
| tail action log std max | 未记录 | 0.3545 |

观察：

- action log std clamp 生效，entropy 和 KL 都明显下降。
- 但策略直接退化为固定 18 步左右倒地。
- 评估标准差很低不是好现象，而是失败行为高度一致。
- `log_std_max=0.5` 对当前任务太保守。

## 本节结论

- action log std 诊断应该保留。
- `action_log_std_max=0.5` 不应作为默认 baseline 配置。
- “压低 entropy / KL” 不能单独作为优化目标，必须同时观察 return 和 episode length。

## 下一节

进入：

- `notes/09_ppo_relaxed_action_std_control.md`

下一节目标：

1. 将 action log std 上限放宽到 `1.0`。
2. 继续使用 `3M` timesteps 长训。
3. 判断较宽松的 std control 能否兼顾探索和稳定。
