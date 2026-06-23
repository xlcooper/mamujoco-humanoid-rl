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

候选方向：

- observation normalization：对 observation 做均值/方差归一化，缓解输入尺度差异，让 actor 和 critic 更容易学习。
- reward/return scaling：对 reward 或 return 做缩放，避免 critic 目标过大或过小，提升 value function 拟合稳定性。
- advantage normalization：对 advantage 做标准化，让 PPO policy update 的梯度尺度更稳定；当前代码已在 PPO update 中使用。
- KL early stopping：监控新旧策略的 approximate KL，超过阈值时提前停止当前 update，防止策略一步改太猛。
- entropy coefficient schedule：动态调整 entropy bonus 系数，控制探索强度从高到低变化，避免早期探索不足或后期过度随机。
- value clipping：像 PPO clip policy 一样限制 value function 更新幅度，减少 critic 在高噪声回报下剧烈震荡。
- orthogonal initialization：用正交初始化网络权重，常见于 PPO 工程实现，可能改善早期训练稳定性。
- vectorized rollout collection：并行采多个环境，提高采样吞吐和 batch 多样性，减少单环境轨迹相关性。
- seed stability comparison：多随机种子验证，判断改进是否稳定，而不是只在单个 seed 上偶然变好。

每个方向都要有明确假设、对照组、指标和结论。

当前状态：

- 正在进行后半段验证。
- 已验证 observation normalization 明显有效，应保留。
- 已验证 KL early stopping 机制有效，但 `target_kl=0.03/0.06` 暂未成为当前主线。
- 已通过长训发现无界 Gaussian policy 会产生严重动作越界。
- 已验证 action log std clamp 不是好主线：`0.5` 和 `1.0` 都导致策略退化。
- 已通过 action clipping diagnostics 发现 tail 中约 `98.26%` raw action 维度被环境裁剪。
- 已引入 tanh-squashed Gaussian policy，并将动作裁剪比例降到 `0`。
- 已通过 `update_epochs=4` 将 squashed policy 的 seed 0 evaluation mean return 提升到 `716.011`。
- 当前正在做 seed `1/2` 多 seed 验证，用于判断 Stage 2 是否可以收束，并准备进入 Stage 3。

### Stage 3：MaMuJoCo 多智能体对比

候选方向：

- `partitioning=None` 单智能体 PPO vs `partitioning="9|8"` 分区控制。
- 参数共享 PPO。
- centralized critic / MAPPO 风格价值函数。
- 对比样本效率、稳定性、最终表现和实现复杂度。

这里会形成更有辨识度的简历亮点，但要等单智能体 PPO 稳定后再做。

进入条件：

- Stage 2 候选单智能体 baseline 至少完成 seed `0/1/2` 验证。
- 有明确的最终单智能体对照配置和复现实验命令。
- 主要失败实验和有效改进已经在 `experiment_records/` 与 `notes/` 中固化。

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

当前正在进行 seed stability comparison。
