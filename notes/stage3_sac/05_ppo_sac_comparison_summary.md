# 05 已完成：PPO/SAC Comparison Summary

## 本节目标

本节基于已有真实结果，先写一版 PPO/SAC 对比总结。

注意：本节不是 SAC 多 seed 最终结论。当前 SAC 只完成 seed `0` 的 `1M` timesteps baseline；SAC seed `1/2` 多 seed 验证暂缓，作为后续补充。

本节回答的问题是：

> 在同一个 MaMuJoCo Humanoid `partitioning=None` 单智能体环境中，当前 SB3 SAC seed `0` 是否形成了明显强于手写 PPO final baseline 的 off-policy 对照？

## 对比对象

### PPO Final Baseline

来源：

- `notes/stage1_2_ppo/13_ppo_squashed_ep4_multiseed.md`
- `experiment_records/ppo_long_obsnorm_squash_ep4_seed0.md`
- `experiment_records/ppo_long_obsnorm_squash_ep4_seed1.md`
- `experiment_records/ppo_long_obsnorm_squash_ep4_seed2.md`

配置：

- 手写 PPO
- observation normalization
- tanh-squashed Gaussian policy
- update epochs: `4`
- total timesteps: `3000000`
- seeds: `0/1/2`

评估结果：

| seed | evaluation mean return | evaluation std | tail rolling episode length |
| --- | ---: | ---: | ---: |
| 0 | 716.011 | 115.490 | 121.450 |
| 1 | 899.806 | 190.234 | 147.175 |
| 2 | 861.418 | 122.380 | 147.185 |

PPO three-seed mean over evaluation means:

```text
825.745
```

### SB3 SAC Seed 0

来源：

- `experiment_records/sac_sb3_1m_seed0.md`
- `notes/stage3_sac/03_sb3_sac_long_training.md`
- `notes/stage3_sac/04_sb3_sac_video_rendering.md`

配置：

- Stable-Baselines3 SAC
- `MlpPolicy`
- `VecNormalize(norm_obs=True, norm_reward=False)`
- automatic entropy tuning: `ent_coef="auto"`
- target entropy: `auto`
- replay buffer size: `1000000`
- learning starts: `10000`
- batch size: `256`
- total timesteps: `1000000`
- seed: `0`

评估结果：

```text
mean_return=6042.360
std_return=37.329
mean_length=1000.000
```

10 个 deterministic evaluation episode 全部达到 `1000` step。

## 定量对比

| 指标 | PPO final seed 0 | PPO final 3-seed mean | SB3 SAC 1M seed 0 |
| --- | ---: | ---: | ---: |
| total timesteps | 3000000 | 3000000 each | 1000000 |
| evaluation mean return | 716.011 | 825.745 | 6042.360 |
| evaluation episode length | 未全部满 1000 | 未全部满 1000 | 1000.000 |
| seeds | 0 | 0/1/2 | 0 |

相对提升：

- SAC seed `0` 相比 PPO seed `0`：约 `8.44x` evaluation mean return。
- SAC seed `0` 相比 PPO three-seed mean：约 `7.32x` evaluation mean return。

这些倍数只用于量化当前实验差距；由于 SAC 还没有多 seed，不能写成多 seed 稳定性结论。

## 行为质量观察

SAC 视频显示策略能稳定站立并持续移动，但姿态明显前倾、屈身，不接近自然人类步态。

这属于 MuJoCo locomotion 中常见的 reward-driven 行为：

- 策略优化的是环境 reward 和存活/前进指标。
- 环境没有直接惩罚“姿态不自然”到足以压过高 return。
- 因此高 return 可以说明任务 reward 表现强，但不能直接说明动作像自然人类走路。

本项目总结时应使用更准确的表述：

- 可以说：SAC 在 reward、episode length 和稳定存活上显著强于当前 PPO baseline。
- 不应说：SAC 学出了自然人形步态。

## 方法层面解释

当前结果符合 Stage 3 的动机：

- PPO 是 on-policy 方法，每次更新主要依赖新 rollout，样本复用效率较低。
- SAC 是 off-policy 方法，通过 replay buffer 复用经验，并用 twin Q critic 和自动熵调节稳定连续控制学习。
- 在 Humanoid 这类高维连续控制任务中，SAC 更容易利用大量历史交互数据改进 critic 和 policy。

本项目中，手写 PPO 的价值不在于打到最高分，而在于展示：

- 从零实现 PPO 训练闭环。
- 通过诊断发现动作越界问题。
- 通过 tanh-squashed policy 和 update epochs tuning 改善训练。
- 完成多 seed 验证和真实消融。

SAC 的价值在于提供成熟 off-policy 强基线，补齐算法对照叙事。

## 当前结论

在当前真实实验结果下：

- Stage 2 手写 PPO final baseline 是一个可复现、可解释、经过多 seed 验证的 on-policy baseline。
- Stage 3 SB3 SAC seed `0` 在 `1M` timesteps 内达到 `6042.360` evaluation mean return，明显高于 PPO final baseline。
- SAC 视频行为稳定但姿态不自然，应如实描述为 reward-driven locomotion。
- SAC multi-seed 暂缓；后续如有时间，优先补 seed `1/2` 来验证稳定性。

## 简历表述建议

可以写：

```text
在手写 PPO 完成多轮诊断与消融后，引入 Stable-Baselines3 SAC 作为 off-policy 强基线；在相同 MaMuJoCo Humanoid 单智能体环境下，SAC seed0 仅用 1M timesteps 即达到 6042.360 evaluation mean return，显著高于 PPO final baseline 的 825.745 三 seed 均值，同时通过视频检查发现策略为高回报但姿态不自然的 reward-driven locomotion。
```

如果需要更保守：

```text
引入 Stable-Baselines3 SAC 作为 off-policy 对照基线，在 seed0 上显著提升 Humanoid 任务 evaluation return 和 episode length；同时结合视频检查区分 reward 表现与自然步态质量，避免只凭数值评估策略行为。
```

## 后续补充

后续可选：

1. 跑 SAC seed `1/2`，形成 SAC 多 seed 稳定性结论。
2. 对 SAC 策略视频做更正式的行为描述。
3. 如果要追求自然步态，考虑 reward shaping、动作平滑惩罚、姿态约束或 imitation learning，但这已经超出当前 Stage 3 主线。
