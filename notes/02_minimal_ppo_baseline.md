# 02 当前任务：普通 PPO 最小训练闭环

## 本节目标

在 `partitioning=None` 的 MaMuJoCo Humanoid 上跑通普通 PPO baseline。

这一节只追求正确、清楚、能训练、能记录日志；暂时不追求最高分，也不做多智能体。

## 环境选择

使用上一节确认通过的单智能体形态：

- MaMuJoCo domain: `Humanoid`
- partitioning: `None`
- agent: `agent_0`
- observation shape: `(348,)`
- action shape: `(17,)`

虽然底层是 PettingZoo Parallel API，但 `partitioning=None` 时只有一个 agent。PPO 代码可以先把它包装成单智能体连续控制环境。

## 本节计划

1. 写一个单智能体环境适配器：从 PettingZoo dict API 转成 PPO 更方便使用的 `(obs, reward, done, info)` 风格。
2. 写 PPO 需要的核心模块：
   - actor-critic network
   - Gaussian continuous policy
   - rollout buffer
   - GAE advantage
   - clipped policy loss
   - value loss
   - entropy bonus
3. 写最小训练入口，先支持短跑：
   - seed
   - total timesteps
   - rollout length
   - minibatch size
   - update epochs
   - TensorBoard 或 CSV 日志
4. 在 AutoDL 上做一次短训练，确认不会崩。

## 第一版不做

- 不做多智能体。
- 不做视频渲染。
- 不做复杂超参搜索。
- 不做论文改进。
- 不直接追求最终高分。

## 预期产出

代码文件候选：

- `src/envs.py`
- `src/ppo.py`
- `src/train_ppo.py`
- `src/evaluate.py`

运行产物放 AutoDL 数据盘，不提交到 Git：

- `/root/autodl-tmp/Humanoid-runs/`
- TensorBoard events
- checkpoints
- evaluation summaries

## 进入代码前还缺什么

最好先把上一节生成的 `server/autodl_host_report.txt` commit/push。这样我可以整理 `AUTODL_HOST_BASELINE.md`，后续实验记录会更完整。

如果你想先写 PPO，也可以直接继续；环境 smoke test 已经足够支持下一步编码。

