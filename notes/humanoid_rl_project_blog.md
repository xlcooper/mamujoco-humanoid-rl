# 从手写 PPO 到 SB3 SAC：MaMuJoCo Humanoid 连续控制项目报告

## 1. 项目背景

这个项目围绕 MaMuJoCo Humanoid 连续控制任务展开，目标是做一个可以完整复盘的强化学习实验项目，而不是只跑通一个现成算法脚本。

Humanoid 是一个比较典型的高维连续控制任务：observation 维度高，action 是连续值，策略需要同时处理身体姿态、关节控制、前进速度和稳定性。相比 CartPole、Pendulum 这类入门环境，Humanoid 更容易暴露强化学习训练中的实际问题，比如：

- critic 拟合压力大；
- policy update 不稳定；
- Gaussian action 超出环境动作边界；
- reward 上升但视频行为并不自然；
- 单个 seed 的结果不足以说明稳定性。

因此，这个项目按阶段推进：

1. 先适配 MaMuJoCo Humanoid 单智能体环境；
2. 手写 PPO baseline，跑通训练和评估闭环；
3. 通过日志诊断和消融优化 PPO；
4. 使用 Stable-Baselines3 SAC 建立 off-policy 强基线；
5. 对 PPO 与 SAC 做阶段性对比，并结合视频观察策略行为。

目前项目已经完成手写 PPO final baseline、PPO 多 seed 验证、SB3 SAC seed0 强基线，以及 PPO/SAC 阶段性对比。SAC 多 seed 验证暂缓，作为后续补充。

## 2. 环境与任务设置

### 2.1 MaMuJoCo Humanoid

MaMuJoCo 来自 Farama Gymnasium-Robotics，底层使用 PettingZoo Parallel API。Humanoid 支持不同的智能体划分方式：

- `partitioning=None`：单智能体控制整个 Humanoid；
- 类似 `9|8` 的 partitioning：把机器人拆成多个智能体。

当前项目先使用 `partitioning=None`，也就是单智能体 Humanoid。这样做的原因是，单智能体设置更接近标准 MuJoCo 连续控制任务，便于先验证 PPO 和 SAC 的训练质量。多智能体分区控制保留为后续扩展方向。

### 2.2 训练与记录方式

训练和评估在 AutoDL Linux 服务器上运行，本地负责代码编辑、Git 管理和结果分析。

项目采用两层产物管理：

```text
/root/autodl-tmp/Humanoid-runs/
```

保存完整运行产物，包括 checkpoint、完整日志、TensorBoard event、视频和评估输出。

```text
experiment_records/
```

保存 Git 管理的轻量实验记录，包括训练配置、指标摘要、evaluation 输出和观察结论。

这样可以避免把大文件提交到 Git，同时又能保留每次实验的关键事实。项目推进过程则写入 `notes/`，每个阶段对应一个 numbered note，一节完成后固化为“已完成总结”。

## 3. Stage 1：手写 PPO 训练闭环

第一阶段先实现一个可读、可调试的 PPO baseline，而不是直接调用成熟库。

主要实现包括：

- 单智能体环境 wrapper：把 MaMuJoCo 的 PettingZoo dict API 转成更适合 PPO 训练的接口；
- Actor-Critic 网络：Gaussian actor 和 value critic；
- rollout buffer；
- GAE；
- PPO clipped objective；
- value loss、entropy logging 和 gradient clipping；
- checkpoint、CSV 日志和 deterministic evaluation。

这一阶段的目标是确认训练闭环可靠：环境能交互，rollout 能采样，PPO update 能执行，checkpoint 能保存和加载，evaluation 能给出可复现结果。

第一条可分析的 PPO baseline v0 在 `100k` timesteps 下得到：

```text
mean_return=243.977
std_return=7.738
```

这个结果不高，但已经有基本学习趋势，可以进入诊断和优化阶段。

## 4. Stage 2：PPO 诊断与优化

PPO 阶段的重点不是单纯堆训练步数，而是根据训练日志逐步定位问题。最终保留下来的主线改动包括 observation normalization、tanh-squashed Gaussian policy 和 update epochs tuning。

### 4.1 Observation normalization

Humanoid observation 维度高，不同维度的数值尺度差异明显。直接输入网络会增加 critic 拟合压力，也会影响 policy update。

项目引入 running mean/std，对 observation 做标准化。短训对比结果：

| 配置 | evaluation mean return |
| --- | ---: |
| PPO baseline v0 | 243.977 |
| PPO + observation normalization | 276.612 |

Observation normalization 让 evaluation mean return 提升约 `13.38%`，value loss 也明显下降。因此该改动被保留为后续 PPO 主线配置。

### 4.2 KL early stopping

加入 observation normalization 后，approximate KL 和 clip fraction 仍然偏高，说明 policy update 幅度偏大。

项目尝试了 target KL early stopping，包括：

- `target_kl=0.03`
- `target_kl=0.06`

这个机制可以限制 update 强度，但 `0.03` 太保守，`0.06` 也没有成为最终最强配置。因此它被记录为有效机制，但没有进入最终 PPO baseline。

### 4.3 长训与动作分布问题

将 observation normalization 配置扩展到 `3M` timesteps 后，evaluation mean return 提升到 `326.992`，说明长训有收益。但同时出现了新的问题：

- entropy 明显升高；
- approximate KL 和 clip fraction 偏高；
- evaluation 波动较大。

一开始尝试限制 Gaussian policy 的 `log_std`，测试 `max=0.5` 和 `max=1.0`，但两个版本都导致策略退化。这说明问题不是简单地压低动作噪声。

随后项目加入 action clipping diagnostics，统计 raw Gaussian action 超出环境动作范围的比例。结果显示，长训 tail 中约 `98.26%` 的 raw action 维度被环境裁剪，平均越界幅度约 `19.55`。

这说明当时的 policy 采样动作与环境实际执行动作严重不一致：PPO 根据 raw action 计算 log probability，但环境实际执行 clipped action，训练目标和真实交互之间出现偏差。

### 4.4 Tanh-squashed Gaussian policy

为了解决动作边界问题，项目将 PPO policy 改成 tanh-squashed Gaussian：

1. 从 Gaussian 分布采样 raw action；
2. 使用 `tanh` 映射到 `[-1, 1]`；
3. 再缩放到环境 action range；
4. 对 log probability 加上 Jacobian correction。

改动后，action clipping fraction 从约 `98.26%` 降到 `0.00%`，策略采样动作和环境执行动作对齐。

### 4.5 Update epochs tuning

初始 squashed policy 仍使用 `update_epochs=10`，KL 和 clip fraction 仍偏高。后续将 update epochs 从 `10` 降到 `4`，降低同一批 rollout 被重复训练的强度。

最终 PPO baseline 配置为：

```bash
python src/train_ppo.py \
  --seed <seed> \
  --total-timesteps 3000000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 4 \
  --run-name ppo_long_obsnorm_squash_ep4_seed<seed> \
  --normalize-observations \
  --squash-actions
```

三 seed 评估结果：

| seed | evaluation mean return | evaluation std | tail action clip fraction |
| --- | ---: | ---: | ---: |
| 0 | 716.011 | 115.490 | 0.0000 |
| 1 | 899.806 | 190.234 | 0.0000 |
| 2 | 861.418 | 122.380 | 0.0000 |

三 seed evaluation mean return：

```text
825.745
```

这个 PPO baseline 不代表最优策略，但它已经满足当前阶段的要求：手写实现、可复现、有诊断、有消融，并完成多 seed 验证。

## 5. Stage 3：引入 SB3 SAC 强基线

PPO 阶段完成后，继续只调 PPO 的收益有限。因此项目引入 Stable-Baselines3 SAC 作为 off-policy 强基线。

引入 SAC 的目的不是替代 PPO 阶段，而是补齐对照：

- PPO 展示手写算法实现、诊断和消融过程；
- SAC 提供成熟 off-policy 方法的强基线；
- 两者共同构成 on-policy 与 off-policy 方法的实验对比。

SAC 使用同一个 `partitioning=None` Humanoid 单智能体环境，并新增 Gymnasium-style wrapper 适配 SB3。

SAC 训练入口默认配置：

- `SAC(MlpPolicy)`；
- `ent_coef="auto"`；
- `target_entropy="auto"`；
- `VecNormalize(norm_obs=True, norm_reward=False)`；
- `Monitor`；
- TensorBoard；
- checkpoint 和 deterministic evaluation。

`5000` timesteps smoke test 确认工程链路后，继续运行 `1M` timesteps seed `0`。

## 6. SAC 1M seed0 结果

SAC seed `0` 的 `1M` timesteps deterministic evaluation 结果：

```text
episode=1 return=6086.414 length=1000
episode=2 return=6043.676 length=1000
episode=3 return=6016.683 length=1000
episode=4 return=6070.044 length=1000
episode=5 return=6017.678 length=1000
episode=6 return=6075.439 length=1000
episode=7 return=6035.541 length=1000
episode=8 return=6097.653 length=1000
episode=9 return=5977.473 length=1000
episode=10 return=6003.000 length=1000
mean_return=6042.360 std_return=37.329
mean_length=1000.000
```

10 个 evaluation episode 全部达到 `1000` step 时间上限。相比 PPO final baseline 的三 seed mean `825.745`，SAC seed `0` 的 reward 表现明显更强。

不过这里需要明确边界：PPO 已完成 seed `0/1/2` 多 seed 验证，SAC 当前只完成 seed `0`。因此当前结论是：SAC seed `0` 已经形成强 off-policy 对照，但 SAC 多 seed 稳定性仍需后续补充。

## 7. 视频观察

SAC seed `0` 的视频显示，策略能够稳定站立并持续移动，不是倒地滑行，也不是评估加载错误。

但视频里的姿态明显前倾、屈身，不接近自然人类步态。也就是说，策略确实学到了能拿高 reward 的 locomotion 行为，但这种行为更偏 reward-driven，而不是自然步态生成。

因此对 SAC 的描述需要分开两层：

- 从 reward 和 episode length 看，SAC seed `0` 显著强于当前 PPO baseline；
- 从视频行为看，SAC 策略稳定但姿态不自然；
- 如果目标是自然步态，还需要额外的 reward 设计、动作平滑、姿态约束或 imitation learning。

这也是为什么项目中保留视频观察：连续控制任务不能只看 return，策略行为本身也需要检查。

## 8. PPO/SAC 阶段性对比

| 指标 | 手写 PPO final | SB3 SAC seed0 |
| --- | ---: | ---: |
| 算法类型 | on-policy | off-policy |
| 实现方式 | 手写 | Stable-Baselines3 |
| 训练步数 | 3M per seed | 1M |
| seeds | 0/1/2 | 0 |
| evaluation mean return | 825.745 三 seed 均值 | 6042.360 |
| mean episode length | 未全部满 1000 | 1000.000 |
| 视频观察 | 分数提升但行为有限 | 稳定移动但姿态不自然 |

阶段性结论：

- 手写 PPO final baseline 是一个可解释、可复现、经过多 seed 验证的 on-policy baseline；
- SB3 SAC seed `0` 在更少 timesteps 下得到显著更高的 reward 和 episode length；
- SAC 当前仍缺少多 seed 验证，不能直接写成多 seed 稳定性结论；
- SAC 视频行为说明高 return 不等于自然步态。

## 9. 当前局限与后续方向

目前已经完成：

- MaMuJoCo Humanoid 单智能体环境适配；
- 手写 PPO baseline；
- PPO 诊断、优化和三 seed 验证；
- SB3 SAC smoke test；
- SB3 SAC `1M` seed0 强基线；
- PPO/SAC 阶段性对比。

仍未完成：

- SAC seed `1/2` 多 seed 验证；
- 更系统的视频行为对比；
- Stage 2 final TensorBoard 补充任务；
- MaMuJoCo 多智能体分区实验。

后续如果继续推进，优先级比较自然的是补 SAC seed `1/2`，验证 seed0 的高分是否稳定。再往后可以扩展到 MaMuJoCo 多智能体分区。如果目标转向更自然的步态，则需要考虑 reward shaping、姿态约束、动作平滑惩罚或 imitation learning。

## 10. 总结

这个项目目前形成了一条比较完整的强化学习实验流程：

1. 先手写 PPO，跑通训练和评估闭环；
2. 通过 observation normalization、action clipping diagnostics、tanh-squashed Gaussian policy 和 update epochs tuning 优化 PPO；
3. 得到三 seed mean return 为 `825.745` 的 PPO final baseline；
4. 引入 SB3 SAC，得到 seed0 `1M` timesteps mean return `6042.360` 的强 off-policy 对照；
5. 通过视频检查补充行为质量判断，避免只用 return 描述策略表现。

当前最稳妥的结论是：

> 手写 PPO final baseline 已完成 seed `0/1/2` 验证，三 seed evaluation mean 为 `825.745`；在同一 MaMuJoCo Humanoid 单智能体环境下，SB3 SAC seed0 用 `1M` timesteps 达到 `6042.360` evaluation mean return，并能稳定跑满 episode，但视频显示策略姿态不自然，更适合描述为 reward-driven locomotion。SAC 多 seed 验证留作后续补充。

