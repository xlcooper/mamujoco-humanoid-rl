# 从手写 PPO 到 SB3 SAC：一次 MaMuJoCo Humanoid 连续控制实验复盘

我最近围绕 MaMuJoCo Humanoid 做了一个连续控制强化学习项目。这个环境的 observation 和 action 维度都比较高，策略一不稳定就很容易表现成摔倒、短暂站立、动作被裁剪或者看起来分数涨了但行为并不合理。

所以这个项目我没有一开始就调用成熟库去跑最高分，而是先手写 PPO，把训练闭环、日志和诊断工具搭起来；等 PPO 有了可以解释的 baseline 之后，再引入 Stable-Baselines3 SAC 做 off-policy 强基线对照。

这篇文章记录目前做到的阶段：环境适配、手写 PPO、PPO 的几轮优化和消融、SB3 SAC seed0 强基线，以及我从视频里看到的一些行为问题。SAC 多 seed 还没有补完，所以这里的 SAC 结论只针对当前 seed0 实验。

## 项目设置

MaMuJoCo 的 Humanoid 环境来自 Farama Gymnasium-Robotics，底层是 PettingZoo Parallel API。这个项目先使用 `partitioning=None`，也就是单智能体 Humanoid。这样做的好处是任务形态更接近标准连续控制，可以先把 PPO 和 SAC 的单智能体实验做清楚，再考虑后续多智能体分区。

项目里我把训练产物和 Git 记录分开管理：

```text
/root/autodl-tmp/Humanoid-runs/
```

这个目录放完整训练产物，比如 checkpoint、TensorBoard event、视频、完整日志。

```text
experiment_records/
```

这个目录只放轻量实验记录，包括配置、指标摘要、评估输出和观察结论。大文件不进 Git，但每次实验的关键结果都能追溯。

这套组织方式是我在项目推进中慢慢固定下来的。强化学习实验很容易散：命令、结果、checkpoint、截图和临时想法混在一起，过几天就很难复盘。把每次实验写成轻量记录之后，后面做对比会轻松很多。

## 先手写 PPO，而不是直接调库

第一阶段我先实现了一个教学友好的 PPO baseline。主要包括：

- 单智能体环境 wrapper，把 MaMuJoCo 的 dict API 转成 PPO 训练更方便的接口。
- Actor-Critic 网络，Gaussian policy 和 value function。
- Rollout buffer。
- GAE。
- PPO clipped objective。
- value loss、entropy、gradient clipping。
- checkpoint、CSV 日志和 deterministic evaluation。

这一阶段的目标不是追求高分，而是确认训练闭环真的可控：能采样、能更新、能保存、能评估，后面出现问题时也知道该去哪里看。

跑通 smoke test 之后，我开始做第一条可以分析的 PPO baseline。

## PPO 第一版：能学，但不稳定

PPO baseline v0 在 `100k` steps 下已经有学习趋势，evaluation mean return 是 `243.977`。这个结果不高，但至少说明训练链路没有问题。

接下来真正的问题变成：怎么让训练更稳定，怎么知道问题出在哪里。

我没有直接加大训练步数，而是先看日志。最明显的问题是 value loss 和 update 指标都不够舒服，说明 critic 和 policy update 都还有优化空间。

## Observation normalization：第一个保留的改动

Humanoid 的 observation 维度很高，不同维度的尺度差异也大。直接把原始 observation 输入网络，critic 会比较吃力。

我加入了 running mean/std，对 observation 做标准化。短训结果从 v0 的 `243.977` 提升到 v1 的 `276.612`，提升约 `13.38%`，value loss 也明显降低。

这个改动比较符合预期，也比较干净，所以后续 PPO 主线都保留了 observation normalization。

## KL early stopping：有用，但不是最终主线

加了 observation normalization 之后，PPO 的 return 上去了，但 approximate KL 和 clip fraction 偏高。也就是说，策略每轮 update 的变化仍然偏猛。

我尝试了 target KL early stopping，测试过 `target_kl=0.03` 和 `target_kl=0.06`。这个机制确实能控制 update 强度，但 `0.03` 太保守，`0.06` 也没有成为最强配置。

所以这部分最后没有进入 final baseline，但它帮我确认了一件事：PPO 当时的主要矛盾确实和 update 强度有关。

## 长训之后，问题转向动作分布

我把 observation normalization 版本拉长到 `3M` timesteps，evaluation mean return 到了 `326.992`。长训确实有提升，但也暴露出新的问题：

- entropy 变得很高。
- approximate KL 和 clip fraction 仍然偏高。
- 策略表现波动明显。

一开始我以为是动作探索噪声太大，于是尝试限制 Gaussian policy 的 `log_std`。结果 `max=0.5` 和 `max=1.0` 都让策略退化了。

这次失败反而很有价值。它说明问题不是简单地“动作太随机”，而是需要更仔细地看动作被环境执行前发生了什么。

## Action clipping diagnostics：真正定位到问题

我给 PPO 加了 action clipping diagnostics，统计 raw Gaussian action 有多少维度超出环境动作范围。

结果很夸张：长训 tail 中约 `98.26%` 的 raw action 维度被环境裁剪，平均越界幅度约 `19.55`。

这意味着 policy 采样出来的动作和环境真正执行的动作严重不一致。PPO 记录 log probability 时认为自己执行的是 raw action，但环境实际执行的是 clipped action。这种错位会让训练信号变得很别扭。

这个诊断基本确定了后面的优化方向：策略本身应该输出合法动作，而不是依赖环境裁剪。

## Tanh-squashed Gaussian policy

我把 PPO 的动作分布改成 tanh-squashed Gaussian：

1. 先从 Gaussian 采样 raw action。
2. 用 `tanh` 压到 `[-1, 1]`。
3. 再映射到环境 action range。
4. 对 log probability 做 Jacobian correction。

改完之后，action clipping fraction 从约 `98.26%` 降到 `0.00%`。这说明策略采样动作和环境执行动作终于对齐了。

不过初始版本的 `update_epochs=10` 仍然让 KL 和 clip fraction 偏高，所以我继续调 update epochs。

## Update epochs 从 10 降到 4

PPO 每次 rollout 会被重复训练多轮。`update_epochs=10` 对当前 squashed policy 来说有点重，于是我把它降到 `4`。

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

三 seed evaluation mean return：

```text
825.745
```

这个 PPO 策略还谈不上完美，视频表现也有限，但它已经是一个比较完整的 on-policy baseline：有手写实现、有诊断、有失败实验、有保留下来的改动，也有多 seed 验证。

## 为什么接着引入 SAC

PPO 做到这里之后，我没有继续在 PPO 上堆更多调参。原因很简单：PPO 已经完成了它在这个项目里的主要任务，也就是展示手写算法和诊断优化过程。

接下来更有价值的是引入一个成熟的 off-policy 强基线，看同一个 Humanoid 任务下，样本复用和自动熵调节能带来多大差异。

所以 Stage 3 我使用 Stable-Baselines3 SAC，而不是再手写 SAC。这样能把精力放在对比实验和结果解释上。

SAC 这边我复用了同一个 Humanoid 单智能体设定，并新增了 Gymnasium 风格 wrapper，让 SB3 可以直接训练。默认配置包括：

- `SAC(MlpPolicy)`
- `ent_coef="auto"`
- `target_entropy="auto"`
- `VecNormalize(norm_obs=True, norm_reward=False)`
- `Monitor`
- TensorBoard
- checkpoint 和 evaluation 输出

先跑 `5000` steps smoke test 确认链路，再跑 `1M` steps seed0。

## SAC 1M seed0：分数差距非常明显

SAC seed0 的 `1M` timesteps 实验结果如下：

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

10 个 evaluation episode 全部跑满 `1000` step。和 PPO final baseline 的三 seed mean `825.745` 相比，SAC seed0 的 reward 表现高了很多。

这里需要留一个边界：PPO 已经做了 seed `0/1/2`，SAC 目前只有 seed `0`。所以我不会把它写成“SAC 多 seed 稳定优于 PPO”，只能说当前 seed0 已经形成了很强的 off-policy 对照。

## 视频检查：分数很高，但姿态不自然

SAC 的视频能看出策略确实在稳定移动，不是倒地滑行，也不是评估脚本加载错了。但它的姿态不自然，身体明显前倾、屈身，看起来不像正常人类步态。

这点我觉得很值得写进总结。因为如果只看 return，很容易把结果说得太漂亮。但视频提醒我：当前策略学到的是对环境 reward 有效的移动方式，不是自然步态。

所以我在总结里会这样表述：

- SAC 在 reward、episode length 和稳定存活上显著强于当前 PPO baseline。
- SAC seed0 视频显示策略能稳定移动，但姿态不自然。
- 这个结果更适合描述为 reward-driven locomotion，而不是自然人形步态。

这也是这个项目里很重要的一课：强化学习实验不能只看曲线和分数。视频、动作分布、环境执行动作和 reward 设计都要一起看。

## PPO 与 SAC 的阶段性对比

| 指标 | 手写 PPO final | SB3 SAC seed0 |
| --- | ---: | ---: |
| 算法类型 | on-policy | off-policy |
| 实现方式 | 手写 | Stable-Baselines3 |
| 训练步数 | 3M per seed | 1M |
| seeds | 0/1/2 | 0 |
| evaluation mean return | 825.745 三 seed均值 | 6042.360 |
| mean episode length | 未全部满 1000 | 1000.000 |
| 视频观察 | 分数提升但行为有限 | 稳定移动但姿态不自然 |

这组对比让我对两个部分的定位更清楚了：

PPO 部分的价值在于“我能把算法写出来，并且能通过诊断和消融把问题讲清楚”。

SAC 部分的价值在于“我能引入成熟 off-policy 强基线，并且用同一个环境做出有说服力的对照”。

这两个价值不冲突。PPO 不需要打过 SAC 才有意义；手写 PPO 的意义在工程理解和诊断过程，SAC 的意义在强基线和方法对照。

## 这次项目里最有价值的几个经验

### 1. 先诊断，再调参

如果没有 action clipping diagnostics，我可能会继续围绕 entropy、log_std、KL 去调很久。但真正的问题是 raw action 和环境执行动作不一致。

诊断工具比盲目调参重要得多。

### 2. 失败实验要保留

Action log std clamp 是失败的，KL early stopping 也没有进入最终配置。但它们都帮助我缩小了问题范围。博客或项目报告里保留这些过程，会比只写“最终用了什么”更像真实实验。

### 3. 手写算法和成熟库不是二选一

手写 PPO 帮我理解了 rollout、GAE、log probability、动作分布和 PPO clipping。SB3 SAC 则让我快速得到强 baseline，避免在手写 SAC 上再花大量工程时间。

项目里两者结合，叙事会比单纯调库更完整。

### 4. 视频检查不能省

SAC 的 return 很高，但视频姿态并不自然。如果不看视频，我可能会把它描述成“学会稳定自然步态”，这就不准确了。

强化学习里的“高分策略”和“人类觉得合理的策略”经常不是一回事。

## 当前局限和后续计划

目前项目已经完成：

- MaMuJoCo Humanoid 单智能体环境适配。
- 手写 PPO baseline。
- PPO 诊断、优化和三 seed 验证。
- SB3 SAC smoke test。
- SB3 SAC `1M` seed0 强基线。
- PPO/SAC 阶段性对比。

还没有完成：

- SAC seed `1/2` 多 seed 验证。
- 更系统的视频行为对比。
- Stage 2 final TensorBoard 补充任务。
- MaMuJoCo 多智能体分区实验。

如果继续推进，我会优先补 SAC seed `1/2`。如果目标从“reward 表现”转向“自然步态”，那就需要考虑 reward shaping、动作平滑惩罚、姿态约束，甚至 imitation learning，这已经是另一个阶段的问题了。

## 总结

到目前为止，这个项目形成了一条比较完整的强化学习实验路径：

先手写 PPO，把训练闭环和诊断工具搭起来；再通过 observation normalization、action clipping diagnostics、tanh-squashed policy 和 update epochs tuning 得到一个可复现的 PPO final baseline；最后引入 SB3 SAC，建立 off-policy 强基线，并通过视频检查补充对策略行为的理解。

当前最稳妥的结论是：

> 手写 PPO final baseline 已完成 seed `0/1/2` 验证，三 seed evaluation mean 为 `825.745`；在同一 MaMuJoCo Humanoid 单智能体环境下，SB3 SAC seed0 用 `1M` timesteps 达到 `6042.360` evaluation mean return，并能稳定跑满 episode，但视频显示策略姿态不自然，更适合描述为 reward-driven locomotion。SAC 多 seed 验证留作后续补充。

这个结果对我来说最有价值的地方，不只是 SAC 分数很高，而是整个过程里每一步都有问题、指标、实验和解释。这样的项目才比较像一个能复盘、能展示、也能继续扩展的强化学习工程项目。

