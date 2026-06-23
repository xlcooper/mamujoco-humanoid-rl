# 00 项目路线图

## 项目定位

这个项目围绕 Farama Gymnasium-Robotics 的 MaMuJoCo Humanoid 环境推进，目标不是只跑一个脚本，而是做成一个可以写进算法工程师简历的强化学习项目：

- 有可复现实验流程。
- 有普通 PPO baseline。
- 有优化或消融对比。
- 有从论文/经典思想中引入的改进点。
- 有最终结论，但结论必须来自真实训练输出。

## 官方环境理解

Farama 文档说明 MaMuJoCo 主要使用 PettingZoo Parallel API；Humanoid 可以用 `partitioning=None` 作为单智能体环境，也可以使用类似 `9|8` 的分区，把机器人拆成多个智能体。

本项目第一阶段先用 `partitioning=None` 跑通普通 PPO。原因是它最接近标准连续控制 PPO，便于检查算法主体、优势估计、策略分布、价值函数和训练循环。多智能体分区放到后续阶段，作为对比和提升空间。

参考：

- <https://robotics.farama.org/envs/MaMuJoCo/ma_humanoid/>
- <https://robotics.farama.org/envs/MaMuJoCo/>

## 阶段规划

### Stage 0：项目启动与环境确认

目标：

- 建好项目管理骨架。
- 采集 AutoDL 主机环境报告。
- 安装依赖并确认 MaMuJoCo Humanoid 可以 reset/step。

产出：

- `README.md`
- `CHANGELOG.md`
- `server/check_autodl_host.sh`
- `src/check_mamujoco_env.py`
- 第一节 lesson note

### Stage 1：普通 PPO baseline

目标：

- 实现或整理一个易读的 PPO 训练闭环。
- 支持连续动作高斯策略、GAE、value loss、entropy bonus、clip objective、evaluation。
- 跑出第一组真实曲线。

建议先只追求正确和可解释，不急着追求最高分。

当前状态：

- 已完成。
- 已实现手写 PPO 最小训练闭环。
- 已完成 PPO baseline v0，并有真实训练、评估和轻量实验记录。
- Stage 1 的作用已经完成：提供一个可运行、可分析、可继续优化的单智能体 PPO 起点。

### Stage 2：工程优化与消融

目标：

- 在普通 PPO baseline 上做工程优化、诊断和消融。
- 每个方向都要有明确假设、对照组、指标和结论。
- 不要求把所有候选方向都做完；只推进能解释当前问题、能形成清晰结论的方向。

已推进方向与当前状态：

1. observation normalization

   中文解释：对 observation 做均值/方差归一化，缓解输入尺度差异，让 actor 和 critic 更容易学习。

   当前状态：已完成。v1 相比 v0 提升 evaluation return，并明显降低 value loss；该方向有效，应保留。

2. advantage normalization

   中文解释：对 advantage 做标准化，让 PPO policy update 的梯度尺度更稳定。

   当前状态：已作为 PPO update 的基础实现保留在代码中，没有单独做消融。

3. KL early stopping

   中文解释：监控新旧策略的 approximate KL，超过阈值时提前停止当前 update，防止策略一步改太猛。

   当前状态：已完成。`target_kl=0.03` 太保守，`target_kl=0.06` 有改善但没有成为当前主线；该机制有效，但暂不作为最终候选 baseline 的核心配置。

4. seed stability comparison

   中文解释：用多个随机种子验证结果是否稳定，避免只在单个 seed 上偶然变好。

   当前状态：已完成。当前候选 baseline 完成 seed `0/1/2`，evaluation mean return 分别为 `716.011`、`899.806`、`861.418`，三 seed mean 为 `825.745`，没有出现 seed 崩塌。

额外推进方向与当前状态：

1. long training

   中文解释：把短训 baseline 扩展到 `3M` timesteps，观察普通 PPO 是否能继续提升，以及长训会暴露什么问题。

   当前状态：已完成。长训提升 return，但暴露 entropy、approx KL、PPO clip fraction 偏高的问题。

2. action log std clamp

   中文解释：限制高斯策略的 `log_std` 范围，尝试控制动作探索噪声。

   当前状态：已完成并判定不作为主线。`max=0.5` 和 `max=1.0` 都导致策略退化，说明硬性 std clamp 不是当前好方案。

3. action clipping diagnostics

   中文解释：统计 raw Gaussian action 有多少维度超出环境动作范围，以及平均越界幅度。

   当前状态：已完成。诊断发现 tail 中约 `98.26%` raw action 维度被环境裁剪，说明无界 Gaussian policy 与环境动作边界严重不匹配。

4. tanh-squashed Gaussian policy

   中文解释：先采样 raw Gaussian action，再用 tanh 映射到环境动作范围，并用 Jacobian 修正 log_prob，让策略天然输出合法动作。

   当前状态：已完成。动作裁剪比例降为 `0`，但初始 `update_epochs=10` 时 KL 和 PPO clip fraction 过高。

5. update epochs tuning

   中文解释：减少同一批 rollout 被重复训练的轮数，降低 PPO update 强度。

   当前状态：已完成。`observation normalization + tanh-squashed Gaussian policy + update_epochs=4` 是 Stage 2 最终单智能体 baseline。

暂未展开方向：

- reward/return scaling
- entropy coefficient schedule
- value clipping
- orthogonal initialization
- vectorized rollout collection

这些方向不是废弃，只是当前问题已经由动作边界诊断、tanh-squashed policy 和 update epochs tuning 得到更直接的推进。Stage 2 已经收束，后续如果 Stage 3 遇到新瓶颈，再按需要回到这些方向。

### Stage 3：MaMuJoCo 多智能体对比

候选方向：

- `partitioning=None` 单智能体 PPO vs `partitioning="9|8"` 分区控制。
- 参数共享 PPO。
- centralized critic / MAPPO 风格价值函数。
- 对比样本效率、稳定性、最终表现和实现复杂度。

这里会形成更有辨识度的简历亮点。当前单智能体 PPO 已经完成 seed `0/1/2` 验证，可以开始进入 Stage 3。

进入条件：

- Stage 2 候选单智能体 baseline 至少完成 seed `0/1/2` 验证：已满足。
- 有明确的最终单智能体对照配置和复现实验命令：已满足。
- 主要失败实验和有效改进已经在 `experiment_records/` 与 `notes/` 中固化：已满足。

当前状态：

- 正在进入 Stage 3。
- 下一步是复核 `partitioning="9|8"` 的多智能体环境接口，并确定第一版多智能体 PPO 路线。

### Stage 4：项目总结与简历材料

目标：

- 固化复现实验命令。
- 写出简历 bullet。
- 整理图表和结论。
- 说明失败实验和调参经验。

## 指标约定

普通 PPO 阶段至少记录：

- episodic return
- episode length
- policy loss
- value loss
- entropy
- approximate KL
- clip fraction
- explained variance
- evaluation return across seeds

多智能体阶段再补充：

- agent-wise reward if available
- global return
- coordination failure cases
- single-agent and multi-agent compute cost

## 第一阶段完成标准

Stage 1 不是“代码写完”就算完成，而是满足：

1. AutoDL 环境报告已记录或至少已贴回关键信息。
2. MaMuJoCo Humanoid smoke test 通过。
3. 普通 PPO 能启动训练并写出日志。
4. 至少完成一次短训练和一次 evaluation。
5. `notes/` 中记录真实现象、问题和下一步。

当前已满足。

## 第二阶段完成标准

Stage 2 不是“某个 seed 分数高”就算完成，而是满足：

1. 明确最终候选单智能体 PPO 配置。
2. 至少完成 seed `0/1/2` 多 seed 验证。
3. 关键优化和失败消融都有真实实验记录。
4. 能解释每个有效改进为什么保留、每个失败方向为什么放弃。
5. README、notes 和 experiment records 能指向一个清晰的最终 baseline。

当前已满足。Stage 2 最终 baseline 见 `notes/13_ppo_squashed_ep4_multiseed.md`。
