# 01 后续任务：Stage 3 SB3 SAC Baseline and Off-Policy Comparison

## 本节目标

Stage 2 已经得到一个稳定但视频表现有限的 PPO baseline。Stage 3 改为引入 SAC，作为更适合 MuJoCo 连续控制的 off-policy 强基线。

这一阶段使用 Stable-Baselines3（SB3），不再手写 SAC。原因是 PPO 阶段已经展示了手写算法能力；SAC 阶段更重要的是快速建立强基线、复用成熟实现，并把精力放在对比实验和结果解释上。

本节的目标不是否定 PPO，而是形成更完整的算法对比：

- PPO：on-policy baseline，训练逻辑清晰，适合展示 GAE、clip objective、KL/clip fraction 诊断。
- SAC：off-policy baseline，样本复用能力更强，通常更适合 Humanoid 这类高维连续控制。

## 为什么 Stage 3 改成 SAC

PPO 已经有充分项目价值：

- 手写实现。
- 真实消融。
- 动作分布诊断。
- 多 seed 验证。

但视频显示当前 PPO 策略主要是短时站立/短时移动，还不是高质量稳定步态。继续只调 PPO 可能会变成低收益调参；引入 SAC 可以更直接地回答：

> 在同一个 Humanoid 环境下，off-policy 方法是否能学出更强、更直观的 locomotion 行为？

这会让项目叙事从“优化 PPO”升级为“比较 on-policy 与 off-policy 连续控制方法”。

## SB3 SAC 主线设计

第一版建议先做 SB3 SAC baseline：

- 使用 SB3 `SAC`。
- 使用 Gymnasium-style 单智能体环境 wrapper。
- 使用 SB3 `Monitor` 记录 episode return/length。
- 使用 SB3 TensorBoard 记录 actor loss、critic loss、entropy coefficient 等曲线。
- 先跑 smoke test，再做 `1M-3M` timesteps 长训。
- 与 PPO final baseline 对比 evaluation return、episode length 和视频行为。

优先目标：

1. 跑通 SB3 SAC 训练入口。
2. 在短训中验证 return、episode length 和 TensorBoard 曲线正常。
3. 进行 `1M-3M` timesteps 长训。
4. 与 PPO final baseline 对比 evaluation return、episode length 和视频行为。

## 优化候选

1. observation normalization

   中文解释：继续处理 Humanoid 高维 observation 尺度差异。

   计划：优先保留为默认选项，但可以做开关消融。

2. automatic entropy tuning

   中文解释：SAC 自动调节探索强度，让策略既探索又不过度随机。

   计划：使用 SB3 SAC 默认 entropy tuning 作为主线。

3. replay buffer size / batch size tuning

   中文解释：off-policy 方法强依赖 replay buffer 和 batch 设置，影响样本复用和训练稳定性。

   计划：先用常见配置，跑通后再调。

4. learning starts

   中文解释：先收集一段随机经验再开始训练，避免最早期 Q 网络在极少数据上过拟合。

   计划：作为基础训练参数保留。

5. target entropy tuning

   中文解释：目标熵控制 SAC 希望策略保留多少随机性。

   计划：先用 `-action_dim` 这类常见设置，后续根据 entropy 和视频表现再调。

## HER 是否适合

HER（Hindsight Experience Replay）主要适合 goal-conditioned sparse reward 任务，例如机械臂抓取、搬运、到达目标位置这类“失败轨迹也能重标目标”的场景。

当前 MaMuJoCo Humanoid 是 dense reward locomotion 任务，不是标准 goal-conditioned 任务。直接上 HER 不自然，除非我们额外把任务改造成“目标速度 / 目标位置 / 目标方向”的 goal-conditioned 版本。

因此：

- HER 不作为 Stage 3 主线。
- 可以在后续扩展中作为“goal-conditioned Humanoid locomotion”的研究型方向。

## 与 PPO 的对比指标

至少对比：

- evaluation mean return
- evaluation std
- episode length
- 视频行为是否更像稳定 locomotion
- 训练 wall-clock 时间
- 样本效率：达到某个 return 阈值所需 timesteps
- 实现复杂度和稳定性

SAC 额外记录：

- actor loss
- critic loss
- Q value mean
- target Q mean
- entropy / log_prob
- alpha
- replay buffer size

## 第一节建议任务

下一节应创建 `notes/stage3_sac/02_sb3_sac_smoke_test.md`，完成：

- 安装并确认 `stable-baselines3` 可用。
- 编写 Gymnasium-compatible Humanoid 单智能体 wrapper，或确认现有 wrapper 是否需要适配。
- 新增 `src/train_sac_sb3.py`。
- 跑通一个很短的 SAC smoke test。
- 确认 TensorBoard 和 evaluation 输出路径。

## 本节完成标准

- 明确 Stage 3 使用 SB3 SAC，不手写 SAC。
- 明确 HER 暂不作为主线。
- 明确下一节从 SB3 SAC smoke test 开始。
