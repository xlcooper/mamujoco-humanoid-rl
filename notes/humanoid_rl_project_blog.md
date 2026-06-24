# 从手写 PPO 到 SB3 SAC：MaMuJoCo Humanoid 连续控制强化学习项目总结

> 当前进度记录时间：2026-06-25  
> 项目环境：Farama Gymnasium-Robotics / MaMuJoCo Humanoid  
> 当前阶段：手写 PPO baseline 与优化已完成，SB3 SAC seed0 强基线与 PPO/SAC 对比总结已完成，SAC 多 seed 验证暂缓作为后续补充。

## 1. 项目背景

这个项目围绕 MaMuJoCo Humanoid 的连续控制任务展开。Humanoid 是一个高维 observation、高维连续 action 的类人机器人 locomotion 环境，比 CartPole、Pendulum 这类入门环境更接近真实强化学习算法工程项目中会遇到的问题：状态尺度复杂、动作边界重要、训练不稳定、评估波动明显，而且策略行为不一定和 reward 数值完全一致。

项目一开始没有直接调用成熟库跑一个结果，而是先手写 PPO baseline，再做诊断、消融和优化；PPO 收束后，再引入 Stable-Baselines3 SAC 作为 off-policy 强基线。这样设计的目的不是单纯追求最高分，而是形成一个完整的实验叙事：

- 从环境适配开始，确认 MaMuJoCo Humanoid 能正常 reset/step。
- 手写一个可读、可复盘的 PPO 训练闭环。
- 用真实训练日志定位问题，而不是凭感觉调参。
- 通过消融实验解释哪些改动有效、哪些方向不适合当前任务。
- 引入 SB3 SAC 对比 on-policy 和 off-policy 方法在 Humanoid 连续控制中的表现差异。

所有实验结论都来自 AutoDL 服务器真实运行结果。大型训练产物、checkpoint、视频和 TensorBoard event 保存在服务器数据盘；Git 仓库只保存代码、notes、轻量 experiment records 和复盘文档。

## 2. 环境与工程组织

MaMuJoCo 使用 PettingZoo Parallel API。当前项目先使用 `partitioning=None`，即单智能体 Humanoid。这样可以把问题简化为标准的连续控制任务，先验证 PPO、SAC 这类单智能体算法的训练质量，再考虑后续多智能体分区控制。

项目采用两层产物管理：

```text
/root/autodl-tmp/Humanoid-runs/
```

保存完整运行产物，例如 `config.json`、完整日志、checkpoint、TensorBoard event、视频等。

```text
experiment_records/
```

保存轻量实验记录，用 Git 管理。每条记录包含训练配置、指标摘要、评估输出和初步观察。

这种组织方式的好处是：训练产物不会污染仓库，但关键实验结论仍然可以通过 Git 追踪。项目推进则由 `notes/` 管理，每一节对应一个明确任务；一节完成后，会改写成“已完成总结”，并新建下一节入口。

## 3. Stage 1：手写 PPO 最小闭环

第一阶段目标是先让训练链路跑通。项目实现了：

- 单智能体 MaMuJoCo wrapper：把 PettingZoo dict API 转成 PPO 更方便使用的单环境接口。
- Actor-Critic 网络：共享 MLP backbone，输出 Gaussian policy 和 value。
- Rollout buffer：保存 observation、action、reward、done、value、log probability。
- GAE：用 generalized advantage estimation 计算 advantage 和 return。
- PPO clipped update：包含 policy loss、value loss、entropy、gradient clipping。
- 训练和评估入口：支持保存 checkpoint、记录 CSV、加载 checkpoint 做 deterministic evaluation。

这一步的核心不是分数，而是工程闭环：环境能交互、PPO 能采样、update 能执行、checkpoint 能保存和加载、evaluation 能跑完。

Smoke test 跑通后，项目进入第一条可分析 baseline。

## 4. Stage 2：PPO 诊断、优化与消融

PPO baseline v0 已经有一定学习趋势，但表现不稳定。后续优化不是直接堆训练步数，而是根据日志逐步定位问题。

### 4.1 Observation normalization

Humanoid observation 维度高，而且不同维度的数值尺度差异很大。直接输入网络会增加 critic 拟合压力，也会影响 actor 更新稳定性。

项目引入 running mean/std，对 observation 做标准化。短训结果显示，observation normalization 将 evaluation mean return 从 v0 的 `243.977` 提升到 v1 的 `276.612`，提升约 `13.38%`；同时 value loss 明显降低，说明 critic 学习压力被缓解。

这个改动被保留为后续 PPO 主线配置。

### 4.2 KL early stopping

PPO v1 的问题是 approximate KL 和 clip fraction 偏高，说明每轮 update 可能过猛。项目加入 target KL early stopping，测试了 `target_kl=0.03` 和 `target_kl=0.06`。

结论是：KL early stopping 机制本身有效，可以压低更新强度；但 `0.03` 太保守，`0.06` 有改善但没有成为最终最强配置。因此它被记录为有效机制，但没有进入最终 baseline 主线。

### 4.3 长训暴露动作探索问题

将 observation normalization 配置扩展到 `3M` timesteps 后，evaluation mean return 提升到 `326.992`，说明长训有帮助。但同时也暴露出新的问题：

- entropy 明显升高。
- approximate KL 和 clip fraction 偏高。
- 策略波动较大。

这提示问题可能不只是 update 强度，还和连续动作分布本身有关。

### 4.4 Action log std clamp 的失败

一个直接想法是限制 Gaussian policy 的 `log_std`，减少动作噪声。项目测试了 `max=0.5` 和 `max=1.0`。

结果并不好：两个版本都导致策略退化。这说明硬性限制标准差不是当前任务的好方案。这个失败实验很重要，因为它把问题从“动作太随机”推进到更具体的问题：采样动作是否和环境动作边界匹配。

### 4.5 Action clipping diagnostics

项目增加了动作裁剪诊断，统计 raw Gaussian action 有多少维度超出环境动作范围。

诊断结果非常明确：长训 tail 中约 `98.26%` 的 raw action 维度被环境裁剪，平均越界幅度约 `19.55`。这意味着 policy 采样出来的动作和环境真正执行的动作严重不一致。PPO 以为自己执行的是 raw action，但环境实际执行的是 clipped action，训练信号和执行行为之间出现偏差。

这个发现是 Stage 2 的关键转折点。

### 4.6 Tanh-squashed Gaussian policy

为了解决动作边界不匹配，项目引入 tanh-squashed Gaussian policy：

1. 先从 Gaussian 分布采样 raw action。
2. 用 `tanh` 把动作压到 `[-1, 1]`。
3. 再线性映射到环境动作范围。
4. 用 Jacobian correction 修正 log probability，保持 PPO 目标函数的数学一致性。

该方案将 action clipping fraction 从约 `98.26%` 降到 `0.00%`。这说明策略输出动作和环境执行动作终于对齐了。

不过初始版本使用 `update_epochs=10`，KL 和 clip fraction 仍偏高。随后项目降低 update epochs。

### 4.7 Update epochs tuning

将 PPO `update_epochs` 从 `10` 降到 `4` 后，squashed policy 的 update 强度明显缓和，evaluation return 提升，同时 action clipping 保持 `0.00%`。

最终 Stage 2 PPO baseline 定为：

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

三 seed 评估结果如下：

| seed | evaluation mean return | evaluation std | tail action clip fraction |
| --- | ---: | ---: | ---: |
| 0 | 716.011 | 115.490 | 0.0000 |
| 1 | 899.806 | 190.234 | 0.0000 |
| 2 | 861.418 | 122.380 | 0.0000 |

三 seed evaluation mean return 的均值为：

```text
825.745
```

这不是完美策略，但它满足了当前阶段的要求：可复现、可解释、有真实消融、有多 seed 验证，可以作为后续 SAC 对照组。

## 5. Stage 3：引入 SB3 SAC 强基线

PPO 阶段已经展示了手写算法、诊断和消融能力。继续只调 PPO 可能会变成低收益调参。因此 Stage 3 改为引入 Stable-Baselines3 SAC，作为 off-policy 强基线。

引入 SAC 的动机是：

- SAC 是 off-policy 方法，可以通过 replay buffer 复用历史经验。
- SAC 使用 twin Q critic 和 stochastic actor，适合连续控制。
- 自动熵调节可以平衡探索和利用。
- Humanoid 这类高维连续控制任务通常更适合用成熟 off-policy 方法建立强基线。

这里不再手写 SAC，因为 PPO 阶段已经覆盖了手写算法能力；SAC 阶段的重点是建立强对照、解释结果和完善项目叙事。

### 5.1 SB3 SAC 工程链路

项目新增了 Gymnasium 风格的 MaMuJoCo 单智能体 wrapper，使 SB3 可以直接使用同一个 `partitioning=None` Humanoid 环境。

SAC 训练入口默认使用：

- `SAC(MlpPolicy)`
- `ent_coef="auto"`
- `target_entropy="auto"`
- `VecNormalize(norm_obs=True, norm_reward=False)`
- `Monitor`
- TensorBoard
- checkpoint、evaluation 和轻量 experiment record

Smoke test `5000` timesteps 跑通后，确认训练、Monitor、TensorBoard、checkpoint、VecNormalize 和 evaluation 链路可用。

### 5.2 SAC 1M seed0 结果

第一条正式 SAC baseline 使用 seed `0`，训练 `1M` timesteps。

配置摘要：

- total timesteps: `1000000`
- learning starts: `10000`
- replay buffer size: `1000000`
- batch size: `256`
- automatic entropy tuning
- observation normalization: SB3 `VecNormalize`

deterministic evaluation 结果：

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

10 个 evaluation episode 全部达到 `1000` step 时间上限。这个结果显著高于 PPO final baseline 的三 seed 均值 `825.745`。

需要注意的是：当前 SAC 只完成了 seed `0`，还不能写成 SAC 多 seed 稳定性结论。SAC seed `1/2` 暂缓，作为后续补充。

## 6. 视频观察：高 return 不等于自然步态

SAC seed0 的视频显示，策略能稳定站立并持续移动，但姿态明显前倾、屈身，不接近自然人类走路。

这个现象在 MuJoCo locomotion 中很常见。策略优化的是环境 reward，不是视觉自然度。如果环境奖励主要鼓励存活和前进，而没有强约束人体运动学美观度，算法可能找到一种高效但不自然的姿态。

因此当前结论应该写得准确：

- 可以说：SAC 在 reward、episode length 和稳定存活上显著强于当前 PPO baseline。
- 不能说：SAC 学出了自然人形步态。
- 更准确的表述是：SAC 学到了高回报但姿态不自然的 reward-driven locomotion。

这也是本项目很重要的经验：强化学习结果不能只看曲线和 return，必须结合视频检查策略行为。数值高并不必然意味着行为符合人的直觉。

## 7. PPO 与 SAC 的阶段性对比

| 指标 | 手写 PPO final | SB3 SAC seed0 |
| --- | ---: | ---: |
| 算法类型 | on-policy | off-policy |
| 实现方式 | 手写 | Stable-Baselines3 |
| 训练步数 | 3M per seed | 1M |
| seeds | 0/1/2 | 0 |
| evaluation mean return | 825.745 三 seed 均值 | 6042.360 |
| mean episode length | 未全部满 1000 | 1000.000 |
| 行为观察 | 分数提升但视频表现有限 | 稳定移动但姿态不自然 |

这组结果说明，在当前 Humanoid 单智能体任务上，SAC seed0 已经形成了明显强于 PPO final baseline 的 off-policy 对照。

但项目也保留了边界：

- PPO 已完成多 seed 验证。
- SAC 当前只有 seed0。
- SAC 高分策略姿态不自然。
- 如果要写最终严谨结论，后续仍应补 SAC seed `1/2`。

## 8. 项目中的关键踩坑与经验

### 8.1 不要只看 return

PPO 中 action clipping diagnostics 证明了这一点：即使 return 在涨，如果 raw action 大量被环境裁剪，策略学习和环境执行之间可能存在严重不一致。

SAC 视频也证明了这一点：即使 return 很高，行为也可能不自然。

### 8.2 失败实验也有价值

Action log std clamp 没有提升表现，但它帮助排除了“简单限制动作噪声”这个方向，使问题定位到动作边界不匹配。

KL early stopping 没有进入最终主线，但它验证了 update control 的有效性，并帮助理解 PPO update 强度。

这些失败实验让项目更像真实工程过程，而不是只展示成功结果。

### 8.3 成熟库和手写算法可以互补

手写 PPO 展示了对算法细节的理解，包括 GAE、clipped objective、log probability、action distribution 和 diagnostics。

SB3 SAC 则提供了成熟强基线，让项目能对比 on-policy 与 off-policy 方法，而不是困在手写实现调参中。

二者结合，比单独“调库跑分”或单独“手写但分数一般”更完整。

## 9. 当前局限与后续方向

当前项目已经完成：

- MaMuJoCo Humanoid 单智能体环境适配。
- 手写 PPO baseline。
- PPO 诊断、优化和多 seed 验证。
- SB3 SAC smoke test。
- SB3 SAC `1M` seed0 强基线。
- PPO/SAC 阶段性对比总结。

仍然存在一些局限：

- SAC 还没有完成 seed `1/2` 多 seed 验证。
- SAC 视频行为不自然，说明 reward 和自然步态之间仍有差距。
- Stage 2 final TensorBoard 补充任务暂缓。
- 多智能体分区控制还没有展开。

后续可以继续做：

1. SAC seed `1/2` 多 seed 验证。
2. 更正式的 PPO/SAC 视频行为对比。
3. MaMuJoCo `partitioning="9|8"` 多智能体扩展。
4. 如果目标转向自然步态，可以考虑 reward shaping、动作平滑惩罚、姿态约束或 imitation learning。

## 10. 总结

这个项目目前形成了一条比较完整的强化学习实验路径：

从手写 PPO 出发，先建立可运行 baseline；再通过 observation normalization、action clipping diagnostics、tanh-squashed policy 和 update epochs tuning 逐步优化；最后引入 SB3 SAC 作为 off-policy 强基线，得到明显更高的 Humanoid reward 表现。

最关键的收获不是某一个分数，而是整个实验方法：

- 每个改动都有假设和指标。
- 每个结果都有轻量记录。
- 每个阶段都能解释为什么继续或放弃。
- 高 return 还会通过视频检查行为质量。

当前最稳妥的项目结论是：

> 手写 PPO final baseline 已完成多 seed 验证，三 seed evaluation mean 为 `825.745`；在同一 MaMuJoCo Humanoid 单智能体环境下，引入 SB3 SAC 后，seed0 在 `1M` timesteps 达到 `6042.360` evaluation mean return，并能稳定跑满 episode，但视频显示其行为更接近 reward-driven locomotion，而不是自然人类步态。SAC 多 seed 验证可作为后续补充。

