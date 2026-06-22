# 01 已完成：MaMuJoCo Humanoid 环境检查

## 结论

本节已完成。MaMuJoCo Humanoid 的无渲染环境检查通过，可以进入下一节普通 PPO 最小训练闭环。

## 已确认

- 远端仓库：`git@github.com:xlcooper/mamujoco-humanoid-rl.git`
- 服务器项目目录：`/root/autodl-tmp/Humanoid`
- 独立 conda 环境：`/root/autodl-tmp/conda-envs/humanoid-rl`
- 已确认包版本：`torch 2.12.1+cu130`、`gymnasium 1.3.0`、`gymnasium_robotics 1.4.2`、`mujoco 3.9.0`、`pettingzoo 1.26.1`、`tensorboard 2.20.0`
- 服务器环境报告文件已在服务器生成：`server/autodl_host_report.txt`，大小约 `6.7K`

注意：当前本地没有拉到 `server/autodl_host_report.txt`。它应该还没有被服务器 commit/push。等这个文件进入 Git 后，再整理 `AUTODL_HOST_BASELINE.md`。

## Smoke Test 结果

### 单智能体：`partitioning=None`

命令：

```bash
python src/check_mamujoco_env.py --partitioning none --steps 5
```

结果：

- `possible_agents=['agent_0']`
- observation space: `Box(shape=(348,), dtype=float64)`
- action space: `Box(shape=(17,), dtype=float32)`
- 5 step 随机 rollout 成功
- total random rollout reward: `24.555850499643142`
- `smoke_test=passed`

这就是第一阶段普通 PPO baseline 要使用的环境形态。

### 多智能体分区：`partitioning="9|8"`

命令：

```bash
python src/check_mamujoco_env.py --partitioning "9|8" --steps 5
```

结果：

- `possible_agents=['agent_0', 'agent_1']`
- `agent_0` observation space: `Box(shape=(242,), dtype=float64)`
- `agent_0` action space: `Box(shape=(9,), dtype=float32)`
- `agent_1` observation space: `Box(shape=(170,), dtype=float64)`
- `agent_1` action space: `Box(shape=(8,), dtype=float32)`
- 5 step 随机 rollout 成功
- total random rollout reward: `{'agent_0': 24.56352757810534, 'agent_1': 24.56352757810534}`
- `smoke_test=passed`

这个结果为后续多智能体对比保留了入口，但下一节先不做多智能体。

## 可以忽略的输出

- `AdroitHand... reward functions were updated...`：Gymnasium-Robotics 关于 Adroit 环境复现的 warning，当前 Humanoid 可以忽略。
- `Error: unable to open display :1`：来自 `glxinfo`，说明没有图形显示窗口；不影响无渲染训练。

## 后续动作

1. 服务器上把环境报告提交并推送：

```bash
git add server/autodl_host_report.txt
git commit -m "Record AutoDL host report"
git pull --rebase
git push
```

2. 进入 `notes/02_minimal_ppo_baseline.md`，开始普通 PPO 最小训练闭环。

