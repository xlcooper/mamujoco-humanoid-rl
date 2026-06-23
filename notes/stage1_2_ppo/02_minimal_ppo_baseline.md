# 02 已完成：普通 PPO 最小训练闭环

## 本节目标

在 `partitioning=None` 的 MaMuJoCo Humanoid 上跑通手写 PPO 的最小训练链路。

本节只验证代码链路：

- 环境适配能工作。
- PPO 能采样 rollout。
- GAE 和 PPO update 能运行。
- checkpoint 能保存。
- evaluate 能加载 checkpoint。

本节不做性能判断。

## 环境形态

使用上一节确认通过的单智能体形态：

- MaMuJoCo domain: `Humanoid`
- partitioning: `None`
- agent: `agent_0`
- observation shape: `(348,)`
- action shape: `(17,)`

## 已完成代码

1. `src/envs.py`
   - 新增 `SingleAgentMaMuJoCoEnv`
   - 将 PettingZoo Parallel API 的 dict observation/action/reward 转成单智能体 PPO loop
   - 在 `step()` 中裁剪动作到环境合法范围，并把 terminated/truncated 合并成 `done`
2. `src/ppo.py`
   - 新增 `ActorCritic`，包含共享 backbone、Gaussian actor 和 critic
   - 新增 `RolloutBuffer`，保存 observation、action、reward、value、log_prob
   - 实现 GAE return/advantage 计算
   - 实现 PPO clipped update、value loss、entropy 和梯度裁剪
3. `src/train_ppo.py`
   - 新增 PPO 训练入口
   - 支持 rollout 采样、GAE 计算、PPO update 和终端日志
   - 保存 `config.json`、`metrics.csv` 和 `checkpoints/agent_final.pt`
4. `src/evaluate.py`
   - 新增 PPO 评估入口
   - 加载 `agent_final.pt`
   - 使用 actor mean action 做确定性评估，并输出 episode return 和 length

代码风格要求见 `README.md`。

## Smoke Test

训练命令：

```bash
cd /root/autodl-tmp/Humanoid
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
python src/train_ppo.py --total-timesteps 4096 --rollout-steps 1024 --batch-size 256 --update-epochs 2 --run-name smoke_ppo
```

评估命令：

```bash
python src/evaluate.py --checkpoint /root/autodl-tmp/Humanoid-runs/smoke_ppo/checkpoints/agent_final.pt --episodes 2
```

实验记录：

- `experiment_records/ppo_smoke_test_001.md`

## 本节结论

- 手写 PPO baseline 的训练入口可以运行到 `training_done=true`。
- `agent_final.pt` 可以被 `src/evaluate.py` 加载并完成短评估。
- 4096 timesteps 太短，只能验证链路，不能作为学习效果或性能结论。

## 下一节

进入：

- `notes/stage1_2_ppo/03_ppo_baseline_v0.md`

下一节目标是跑第一条可分析 PPO baseline，并通过 `experiment_records/` Git 管理结果摘要。
