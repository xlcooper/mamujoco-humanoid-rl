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

### Stage 2：工程优化与消融

候选方向：

- observation normalization
- reward/return scaling
- advantage normalization
- KL early stopping
- entropy coefficient schedule
- value clipping
- orthogonal initialization
- vectorized rollout collection
- seed stability comparison

每个方向都要有明确假设、对照组、指标和结论。

### Stage 3：MaMuJoCo 多智能体对比

候选方向：

- `partitioning=None` 单智能体 PPO vs `partitioning="9|8"` 分区控制。
- 参数共享 PPO。
- centralized critic / MAPPO 风格价值函数。
- 对比样本效率、稳定性、最终表现和实现复杂度。

这里会形成更有辨识度的简历亮点，但要等单智能体 PPO 稳定后再做。

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

