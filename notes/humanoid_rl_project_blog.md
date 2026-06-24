# 从手写 PPO 到 SB3 SAC：MaMuJoCo Humanoid 连续控制实验总结

这篇文章记录一个 MaMuJoCo Humanoid 连续控制项目目前的阶段性结果。项目的目标不是只跑通一个强化学习脚本，而是从环境适配开始，逐步完成手写 PPO baseline、训练诊断、消融优化，再引入 SB3 SAC 作为 off-policy 强基线，形成一条可以复盘的实验流程。

当前使用的是 MaMuJoCo Humanoid 的 `partitioning=None` 单智能体设置。这个任务的 observation 维度高，action 是连续控制，训练过程中很容易遇到 value loss 偏高、策略更新过猛、动作越界、视频行为和 reward 不一致等问题。因此，项目的重点放在“怎么发现问题”和“怎么解释改动”上，而不只是最终分数。

## 项目组织

训练和评估统一在 AutoDL Linux 服务器上运行，本地负责代码编辑、Git 管理和结果分析。运行产物和 Git 记录分开保存：

```text
/root/autodl-tmp/Humanoid-runs/
```

这里保存完整训练产物，包括 checkpoint、完整日志、TensorBoard event、视频和评估输出。

```text
experiment_records/
```

这里保存轻量实验记录，包括配置、关键指标、评估结果和观察结论。完整 run 目录不提交到 Git，但每次实验都保留足够复盘的信息。

项目笔记放在 `notes/` 下，每个阶段用 numbered note 管理。一节完成后会改成“已完成总结”，下一步再新建新的 note。这样做的好处是，实验推进过程不会只停留在聊天记录或临时命令里。

## 手写 PPO baseline

项目第一阶段先手写 PPO。核心模块包括：

- 单智能体 MaMuJoCo wrapper，把 PettingZoo Parallel API 转成更适合 PPO 训练的接口。
- Actor-Critic 网络，包含 Gaussian policy 和 value function。
- Rollout buffer 和 GAE。
- PPO clipped objective、value loss、entropy logging 和 gradient clipping。
- 训练入口、评估入口、checkpoint 和 CSV 日志。

这一阶段主要是确认训练闭环可靠：环境能交互，rollout 能采样，PPO update 能执行，checkpoint 能保存和加载，evaluation 能得到可复现输出。

第一条可分析的 PPO baseline v0 在 `100k` timesteps 下 evaluation mean return 为 `243.977`。这个分数不高，但已经能看到基本学习趋势，说明后续可以开始做诊断和优化。

## Observation normalization

Humanoid 的 observation 维度和尺度差异都比较大，直接输入网络会增加 critic 拟合压力。项目引入 running mean/std，对 observation 做标准化。

加入 observation normalization 后，短训 evaluation mean return 从 `243.977` 提升到 `276.612`，提升约 `13.38%`，value loss 也明显下降。这说明输入尺度处理对当前 PPO baseline 有帮助，因此后续 PPO 主线都保留了这个配置。

## PPO update control

Observation normalization 提升了回报，但 approximate KL 和 clip fraction 仍然偏高，说明 policy update 幅度偏大。项目尝试了 target KL early stopping，包括 `target_kl=0.03` 和 `target_kl=0.06`。

这个机制能限制 update 强度，但 `0.03` 太保守，`0.06` 也没有成为最终最强配置。因此 KL early stopping 被记录为有效机制，但没有进入最终 baseline 的核心配置。

## 长训后的动作分布问题

将 observation normalization 配置扩展到 `3M` timesteps 后，evaluation mean return 提升到 `326.992`，说明长训有收益。但同时出现了新的问题：

- entropy 明显升高；
- approximate KL 和 clip fraction 偏高；
- evaluation 波动较大。

项目先尝试限制 Gaussian policy 的 `log_std`，测试了 `max=0.5` 和 `max=1.0`。两个版本都导致策略退化，说明简单压低动作噪声并不是合适方向。

随后加入 action clipping diagnostics，统计 raw Gaussian action 有多少维度超出环境动作范围。诊断结果显示，长训 tail 中约 `98.26%` 的 raw action 维度被环境裁剪，平均越界幅度约 `19.55`。

这个结果说明，当时的 policy 采样动作和环境实际执行动作严重不一致。PPO 根据 raw action 计算 log probability，但环境执行的是 clipped action，训练目标和真实交互之间出现偏差。

## Tanh-squashed Gaussian policy

为了解决动作边界问题，项目将 PPO policy 改成 tanh-squashed Gaussian：

1. 从 Gaussian 分布采样 raw action；
2. 通过 `tanh` 映射到 `[-1, 1]`；
3. 再缩放到环境 action range；
4. 对 log probability 加上 Jacobian correction。

这个改动把 action clipping fraction 从约 `98.26%` 降到 `0.00%`，动作采样和环境执行终于对齐。

初始 squashed policy 仍使用 `update_epochs=10`，KL 和 clip fraction 还是偏高。继续调低 update 强度后，最终将 `update_epochs` 从 `10` 降到 `4`。

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

三 seed 评估结果如下：

| seed | evaluation mean return | evaluation std | tail action clip fraction |
| --- | ---: | ---: | ---: |
| 0 | 716.011 | 115.490 | 0.0000 |
| 1 | 899.806 | 190.234 | 0.0000 |
| 2 | 861.418 | 122.380 | 0.0000 |

三 seed evaluation mean return 为：

```text
825.745
```

这个 PPO baseline 不算高分策略，但它具备几个重要特点：实现是手写的，关键问题有诊断过程，有失败消融，也完成了 seed `0/1/2` 验证。

## 引入 SB3 SAC

PPO 阶段已经完成了手写算法和优化诊断的主要目标。继续只调 PPO 的边际收益不高，因此下一步引入 Stable-Baselines3 SAC 作为 off-policy 强基线。

SAC 的引入主要用于回答另一个问题：在同一个 Humanoid 单智能体环境中，off-policy 样本复用、twin Q critic 和自动熵调节，是否能比当前 on-policy PPO baseline 学到更强的策略。

SAC 训练入口使用：

- `SAC(MlpPolicy)`；
- `ent_coef="auto"`；
- `target_entropy="auto"`；
- `VecNormalize(norm_obs=True, norm_reward=False)`；
- `Monitor` 和 TensorBoard；
- checkpoint、evaluation 和轻量实验记录。

`5000` timesteps smoke test 跑通后，项目继续运行 `1M` timesteps seed `0`。

## SAC 1M seed0 结果

SAC seed `0` 的 `1M` timesteps deterministic evaluation 结果如下：

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

这里需要保留实验边界：PPO 已经完成三 seed 验证，而 SAC 当前只完成 seed `0`。因此当前结论是“SAC seed0 形成了很强的 off-policy 对照”，不是“SAC 多 seed 稳定优于 PPO”。

## 视频观察

SAC seed `0` 的视频显示，策略可以稳定站立并持续移动，没有表现为倒地后滑行。但姿态明显前倾、屈身，不接近自然人类步态。

这对结果解释很重要。单看 return，SAC 表现非常强；结合视频看，它更像是学到了对当前 reward 有效的移动方式，而不是自然步态。

因此，当前更准确的描述是：

- SAC 在 reward、episode length 和稳定存活上显著强于当前 PPO baseline；
- SAC seed0 策略视频表现为高回报但姿态不自然的 locomotion；
- 如果目标是自然步态，还需要额外的 reward 设计、姿态约束、动作平滑约束或 imitation learning。

## PPO 与 SAC 阶段性对比

| 指标 | 手写 PPO final | SB3 SAC seed0 |
| --- | ---: | ---: |
| 算法类型 | on-policy | off-policy |
| 实现方式 | 手写 | Stable-Baselines3 |
| 训练步数 | 3M per seed | 1M |
| seeds | 0/1/2 | 0 |
| evaluation mean return | 825.745 三 seed 均值 | 6042.360 |
| mean episode length | 未全部满 1000 | 1000.000 |
| 视频观察 | 分数提升但行为有限 | 稳定移动但姿态不自然 |

PPO 的价值主要体现在手写实现、诊断和消融过程上。它提供了一个可解释的 on-policy baseline，也暴露了连续动作策略中动作边界处理的重要性。

SAC 的价值在于提供成熟 off-policy 强基线。seed `0` 的结果已经说明，在当前 Humanoid 单智能体任务上，SAC 能以更少 timesteps 获得明显更高的 reward 表现。

这两部分并不是互相替代的关系。PPO 部分展示算法理解和工程诊断，SAC 部分补齐强基线和方法对照。

## 当前局限

当前项目已经完成：

- MaMuJoCo Humanoid 单智能体环境适配；
- 手写 PPO baseline；
- PPO 诊断、优化和三 seed 验证；
- SB3 SAC smoke test；
- SB3 SAC `1M` seed0 强基线；
- PPO/SAC 阶段性对比。

目前还没有完成：

- SAC seed `1/2` 多 seed 验证；
- 更系统的视频行为对比；
- Stage 2 final TensorBoard 补充任务；
- MaMuJoCo 多智能体分区实验。

如果后续继续推进，优先级比较自然的是补 SAC seed `1/2`，验证当前 seed0 结果是否稳定。再往后可以考虑 MaMuJoCo 多智能体分区，或者如果目标转向自然步态，再考虑 reward shaping 和 imitation learning。

## 小结

目前这个项目已经形成了比较完整的一轮强化学习实验闭环：

1. 先手写 PPO，跑通训练和评估；
2. 通过日志诊断发现 observation 尺度、update 强度和动作边界问题；
3. 通过 observation normalization、tanh-squashed Gaussian policy 和 update epochs tuning 得到 PPO final baseline；
4. 引入 SB3 SAC，建立 off-policy 强基线；
5. 结合 evaluation 和视频观察，区分 reward 表现和行为自然度。

当前最稳妥的阶段性结论是：

> 手写 PPO final baseline 已完成 seed `0/1/2` 验证，三 seed evaluation mean 为 `825.745`；在同一 MaMuJoCo Humanoid 单智能体环境下，SB3 SAC seed0 用 `1M` timesteps 达到 `6042.360` evaluation mean return，并能稳定跑满 episode，但视频显示策略姿态不自然，更适合描述为 reward-driven locomotion。SAC 多 seed 验证留作后续补充。

