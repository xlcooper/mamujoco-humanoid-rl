# 04 当前任务：PPO 诊断与 Observation Normalization

## 本节目标

基于 `ppo_baseline_v0_seed0` 的真实结果，增强 PPO baseline 的稳定性和可分析性。

本节不追求最终高分，目标是让下一次实验更标准、更容易判断问题来源。

## 为什么做这一节

baseline v0 已经说明手写 PPO 能学到一些行为，但也暴露出几个现象：

- episode return 波动仍然很大。
- critic 的 value loss 仍偏高。
- Humanoid observation 维度为 `348`，不同观测量尺度差异可能很大。
- PPO 的 `approx_kl` 和 `clip_fraction` 偏高，后续需要更仔细观察更新幅度。

连续控制任务里，observation normalization 是非常常见且基础的稳定性组件。先补它，比直接把训练拉长到数百万步更稳。

## 已完成代码任务

1. 已增加 running mean/std 工具：
   - 跟踪 observation 均值和方差。
   - 训练时更新统计量。
   - 输入网络前归一化 observation。
2. 已保存 normalization 状态：
   - checkpoint 需要保存 actor-critic 参数和 observation 统计量。
   - evaluate 加载 checkpoint 时使用同一套统计量。
3. 已改善日志：
   - 增加 rolling episode return。
   - 增加 rolling episode length。
   - 保留现有 PPO 指标：policy loss、value loss、entropy、approx KL、clip fraction。

相关代码：

- `src/normalization.py`
- `src/train_ppo.py`
- `src/evaluate.py`

## 本节实验任务

现在运行 baseline v1：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/train_ppo.py \
  --total-timesteps 100000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 10 \
  --run-name ppo_baseline_v1_obsnorm_seed0 \
  --normalize-observations
```

评估：

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_baseline_v1_obsnorm_seed0/checkpoints/agent_final.pt \
  --episodes 5 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_baseline_v1_obsnorm_seed0/eval_output.txt
```

生成 Git 管理的实验记录：

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_baseline_v1_obsnorm_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_baseline_v1_obsnorm_seed0/eval_output.txt \
  --output experiment_records/ppo_baseline_v1_obsnorm_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_baseline_v1_obsnorm_seed0.md
git commit -m "Record PPO baseline v1 obs norm seed0 summary"
git pull --rebase
git push
```

## 对比重点

和 `experiment_records/ppo_baseline_v0_seed0.md` 对比：

- evaluation mean return 是否提高。
- episode length 是否更稳定。
- value loss 是否下降或波动变小。
- approx KL 和 clip fraction 是否更可控。
- entropy 是否仍没有塌缩。

## 本节完成标准

- observation normalization 代码完成并有中文注释。
- `ppo_baseline_v1_obsnorm_seed0` 实验记录通过 Git 推回。
- 根据 v0 vs v1 判断下一节方向：
  - 如果明显改善：进入更长训练和多 seed。
  - 如果改善不明显：检查 reward scaling、learning rate、update epochs 或 KL early stopping。
