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

## 已完成

- 上一节生成的 `server/autodl_host_report.txt` 已经进入 Git，稳定环境事实已整理到 `AUTODL_HOST_BASELINE.md`。
- 手写 PPO baseline 代码已完成第一版。
- Smoke test 已通过，结果已整理到 `experiment_records/ppo_smoke_test_001.md`。

## 当前任务：PPO baseline v0

现在要跑一个中等长度 baseline，用来获得第一条可分析训练曲线。它仍然不是最终成绩，只用于判断当前 PPO 实现是否有学习趋势、是否有明显稳定性问题。

在 AutoDL 上运行：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/train_ppo.py \
  --total-timesteps 100000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 10 \
  --run-name ppo_baseline_v0_seed0
```

训练结束后评估：

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_baseline_v0_seed0/checkpoints/agent_final.pt \
  --episodes 5
```

检查轻量输出：

```bash
tail -n 20 /root/autodl-tmp/Humanoid-runs/ppo_baseline_v0_seed0/metrics.csv
cat /root/autodl-tmp/Humanoid-runs/ppo_baseline_v0_seed0/config.json
```

## 你需要回传

把下面内容贴回来，之后整理到 `experiment_records/ppo_baseline_v0_seed0.md`：

1. 训练最后 20 行 `metrics.csv`。
2. `config.json`。
3. evaluate 输出。
4. 如果报错，贴 traceback 最后 80 行。

## 分析重点

看这几个现象：

- episodic return 是否有上升趋势。
- episode length 是否变长。
- value loss 是否异常爆炸。
- entropy 是否过快下降。
- approx KL 和 clip fraction 是否显示更新过猛。

如果 baseline 明显不稳定，下一步优先考虑 observation normalization / reward scaling。若曲线能正常上升，再进入更长训练和多 seed。
