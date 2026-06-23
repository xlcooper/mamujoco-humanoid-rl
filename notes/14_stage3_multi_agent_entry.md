# 14 当前任务：Stage 3 多智能体入口检查

## 本节目标

Stage 2 已经得到稳定的单智能体 PPO baseline。本节开始进入 Stage 3：MaMuJoCo Humanoid 多智能体对比。

本节不急着直接训练多智能体 PPO，而是先确认多智能体分区环境的接口、观测/动作空间、reward 形态和代码改造范围。

## 背景

当前单智能体对照组：

- environment: MaMuJoCo Humanoid, `partitioning=None`
- algorithm: hand-written PPO
- observation normalization: on
- policy: tanh-squashed Gaussian
- update epochs: `4`
- total timesteps: `3000000`
- seeds: `0/1/2`
- evaluation mean over seed means: `825.745`

这个结果将作为 Stage 3 的单智能体对照。

## 为什么先做入口检查

多智能体不是简单把 PPO 训练脚本复制一份：

- `partitioning="9|8"` 会出现多个 agent，每个 agent 有自己的 observation/action。
- reward 可能是共享的，也可能按 agent 返回，需要确认实际 API。
- 单智能体 `SingleAgentMaMuJoCoEnv` 只取 `agent_0`，不能直接用于多智能体训练。
- 参数共享 PPO、独立 PPO、centralized critic / MAPPO 的数据结构不同，先看清接口再写代码更稳。

## 在服务器运行接口检查

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/check_mamujoco_env.py --partitioning "9|8" --steps 5 \
  | tee /root/autodl-tmp/Humanoid-runs/mamujoco_humanoid_9x8_check.txt
```

如果需要再补一个单智能体对照输出：

```bash
python src/check_mamujoco_env.py --partitioning none --steps 5 \
  | tee /root/autodl-tmp/Humanoid-runs/mamujoco_humanoid_single_check.txt
```

## 回传轻量记录

服务器输出不用整段贴聊天。把关键信息写入一个轻量记录：

```bash
mkdir -p experiment_records
nano experiment_records/mamujoco_humanoid_9x8_env_check.md
```

建议记录：

- possible agents
- active agents
- 每个 agent 的 observation space
- 每个 agent 的 action space
- step 后 rewards 的 key 和数值形态
- 是否 smoke_test=passed
- 和 `partitioning=None` 的主要区别

提交：

```bash
git add experiment_records/mamujoco_humanoid_9x8_env_check.md
git commit -m "Record MaMuJoCo Humanoid multi-agent env check"
git pull --rebase
git push
```

## 本地分析重点

拿到记录后，本地需要判断：

1. `9|8` 分区下有几个 agent。
2. 每个 agent 的 observation/action 维度是否相同。
3. reward 是每个 agent 同值、局部值，还是别的结构。
4. 参数共享 PPO 是否可以作为 Stage 3 的第一版。
5. centralized critic 是否需要在第一版就做，还是先作为后续优化。

## 本节完成标准

- `partitioning="9|8"` smoke test 记录进入 Git。
- 明确 Stage 3 第一版算法路线。
- 新建下一节 note，进入多智能体 PPO 代码实现或更细的算法设计。
