# PPO Smoke Test 001

## 目的

验证手写 PPO baseline 的最小训练链路是否能在 AutoDL 上跑通。

本记录只说明代码链路可运行，不作为 Humanoid PPO 性能结论。

## 环境

- 环境基线：见 `AUTODL_HOST_BASELINE.md`
- MaMuJoCo domain: `Humanoid`
- partitioning: `None`
- agent: `agent_0`
- observation shape: `(348,)`
- action shape: `(17,)`
- run directory: `/root/autodl-tmp/Humanoid-runs/smoke_ppo`

## 训练命令

```bash
cd /root/autodl-tmp/Humanoid
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
python src/train_ppo.py --total-timesteps 4096 --rollout-steps 1024 --batch-size 256 --update-epochs 2 --run-name smoke_ppo
```

## 训练输出摘要

| update | global_step | last_ep_return | last_ep_len | mean_reward | policy_loss | value_loss | entropy | approx_kl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 79.886 | 18 | 4.459 | -0.0257 | 757.2495 | 24.1197 | 0.021101 |
| 2 | 2048 | 77.160 | 18 | 4.492 | -0.0076 | 811.2772 | 24.1206 | 0.009814 |
| 3 | 3072 | 89.061 | 20 | 4.427 | -0.0063 | 730.2310 | 24.1205 | 0.010746 |
| 4 | 4096 | 111.467 | 27 | 4.427 | -0.0037 | 720.7236 | 24.1214 | 0.010707 |

训练结束标志：

```text
training_done=true run_dir=/root/autodl-tmp/Humanoid-runs/smoke_ppo
```

## 评估命令

```bash
python src/evaluate.py --checkpoint /root/autodl-tmp/Humanoid-runs/smoke_ppo/checkpoints/agent_final.pt --episodes 2
```

## 评估输出摘要

| episode | return | length |
| --- | ---: | ---: |
| 1 | 124.564 | 31 |
| 2 | 137.026 | 34 |

```text
mean_return=130.795 std_return=6.231
```

## 观察

- 训练入口、checkpoint 保存、checkpoint 加载和评估入口均可运行。
- `AdroitHand... reward functions were updated...` warning 与当前 Humanoid 任务无关，暂时忽略。
- 4096 timesteps 太短，只能验证链路，不足以判断 PPO 学习效果。

## 下一步

- 检查服务器 run 目录中的 `config.json` 和 `metrics.csv`。
- 跑一个更长的 baseline，再开始讨论训练稳定性和优化项。
