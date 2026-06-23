# 09 当前任务：PPO Relaxed Action Std Control

## 本节目标

继续验证 action std control，但把上限从 `0.5` 放宽到 `1.0`。

08 说明 `log_std_max=0.5` 能压低 entropy 和 KL，但会把策略压到固定 18 步倒地。这个实验不能证明“std control 无效”，只能证明 `0.5` 太紧。

本节做更温和的对比：`action_log_std_max=1.0`。

## 为什么不是直接放弃 std control

07 的长训基线虽然最好，但存在明显风险：

- evaluation std: `80.387`
- tail entropy mean: `54.67`
- tail approx KL mean: `0.2990`
- tail clip fraction mean: `0.5571`

这些指标说明训练过程仍然不稳定。08 的失败告诉我们不能用太紧的 clamp，但不代表完全不控制探索就是最终答案。

## 本节实验设计

主实验：

- run name: `ppo_long_obsnorm_logstd10_seed0`
- total timesteps: `3000000`
- normalize observations: `true`
- target KL: 不启用
- action log std max: `1.0`
- action log std min: `-5.0`
- seed: `0`

`log_std=1.0` 对应动作标准差约 `exp(1.0)=2.72`，比 `0.5` 宽松很多，但仍低于 07 长训中推测出的极高探索强度。

## 在服务器运行训练

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/train_ppo.py \
  --total-timesteps 3000000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 10 \
  --run-name ppo_long_obsnorm_logstd10_seed0 \
  --normalize-observations \
  --action-log-std-min -5.0 \
  --action-log-std-max 1.0
```

## 训练完成后评估

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_logstd10_seed0/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_logstd10_seed0/eval_output.txt
```

## 生成 Git 管理的实验记录

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_logstd10_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_logstd10_seed0/eval_output.txt \
  --output experiment_records/ppo_long_obsnorm_logstd10_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_long_obsnorm_logstd10_seed0.md
git commit -m "Record PPO long obs norm log std 10 seed0 summary"
git pull --rebase
git push
```

## 对比重点

同时对比 07 和 08：

- 是否明显超过 `logstd05` 的 `78.767`。
- 是否接近或超过 07 的 `326.992`。
- evaluation std 是否低于 07 的 `80.387`。
- entropy 是否处在 `20.52` 和 `54.67` 之间。
- approx KL 是否低于 07 的 `0.2990`。
- clip fraction 是否低于 07 的 `0.5571`。
- action log std max 是否接近或触碰 `1.0`。

## 本节完成标准

- `ppo_long_obsnorm_logstd10_seed0` 完成长训。
- `experiment_records/ppo_long_obsnorm_logstd10_seed0.md` 被 Git 管理并推回。
- 根据结果判断：
  - 如果表现接近 07 且指标更稳：将 `log_std_max=1.0` 作为候选 baseline。
  - 如果仍明显退化：放弃硬 clamp，进入 action clipping diagnostics 或 squashed Gaussian policy。
