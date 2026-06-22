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

1. 已写一个单智能体环境适配器：从 PettingZoo dict API 转成 PPO 更方便使用的 `(obs, reward, done, info)` 风格。
2. 已写 PPO 需要的核心模块：
   - actor-critic network
   - Gaussian continuous policy
   - rollout buffer
   - GAE advantage
   - clipped policy loss
   - value loss
   - entropy bonus
3. 已写最小训练入口，先支持短跑：
   - seed
   - total timesteps
   - rollout length
   - minibatch size
   - update epochs
   - CSV 日志
4. 已在 AutoDL 上完成一次短训练和短评估，确认训练链路可以跑通。

## 第一版不做

- 不做多智能体。
- 不做视频渲染。
- 不做复杂超参搜索。
- 不做论文改进。
- 不直接追求最终高分。

## 预期产出

已新增代码文件：

- `src/envs.py`
- `src/ppo.py`
- `src/train_ppo.py`
- `src/evaluate.py`

运行产物放 AutoDL 数据盘，不提交到 Git：

- `/root/autodl-tmp/Humanoid-runs/`
- `config.json`
- `metrics.csv`
- checkpoints
- evaluation summaries

## 进入代码前还缺什么

上一节生成的 `server/autodl_host_report.txt` 已经进入 Git，稳定环境事实已整理到 `AUTODL_HOST_BASELINE.md`。

短训练已通过。运行命令：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
python src/train_ppo.py --total-timesteps 4096 --rollout-steps 1024 --batch-size 256 --update-epochs 2 --run-name smoke_ppo
```

如果短训练通过，再运行评估：

```bash
python src/evaluate.py --checkpoint /root/autodl-tmp/Humanoid-runs/smoke_ppo/checkpoints/agent_final.pt --episodes 2
```

把训练输出和是否生成 `metrics.csv` 贴回来。短训练结果只用于检查代码链路，不作为性能结论。

结果已经整理到 `experiment_records/ppo_smoke_test_001.md`。

## 当前结论

- 手写 PPO baseline 的训练入口可以运行到 `training_done=true`。
- `agent_final.pt` 可以被 `src/evaluate.py` 加载并完成 2 episode 评估。
- 这次只是 smoke test，不作为算法性能结论。

## 下一步

下一节建议做一次更完整的短基线：

1. 确认 `metrics.csv` 和 `config.json` 的内容。
2. 跑一个更长但仍可控的 baseline，例如 `100_000` 到 `300_000` timesteps。
3. 根据训练曲线决定是否先加 observation normalization / reward scaling。
