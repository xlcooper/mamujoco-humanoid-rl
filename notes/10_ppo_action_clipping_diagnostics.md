# 10 当前任务：PPO Action Clipping Diagnostics

## 本节目标

诊断连续动作高斯策略采样出来的 raw action 有多少被环境动作边界裁剪。

07 的最强长训存在 entropy、KL、clip fraction 过高的问题；08 和 09 说明硬性限制 action log std 会让策略退化。现在更合理的怀疑是：

> 策略采样的 raw Gaussian action 经常超出环境动作范围，`envs.py` 会把它裁剪后送入 MuJoCo，但 PPO 计算 log_prob 时仍使用未裁剪的 raw action。

如果裁剪比例很高，就会产生训练信号和真实执行动作不一致的问题。这时下一步应考虑 tanh-squashed Gaussian policy 或动作分布缩放，而不是继续调 `log_std_max`。

## 本节代码变化

已在 `src/train_ppo.py` 增加 rollout 级动作裁剪诊断：

- `action_clip_fraction`
  - 每个 rollout 中，raw action 维度超出环境动作边界的平均比例。
- `action_clip_excess_mean`
  - 超出边界的平均幅度；未超出边界的维度记为 `0`。

这两个指标只用于诊断，不改变训练行为。

## 本节实验设计

主实验：

- run name: `ppo_long_obsnorm_clipdiag_seed0`
- total timesteps: `3000000`
- normalize observations: `true`
- target KL: 不启用
- action log std clamp: 不启用
- seed: `0`

这个配置等价于当前最强的 07，只是多记录动作裁剪诊断指标。

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
  --run-name ppo_long_obsnorm_clipdiag_seed0 \
  --normalize-observations
```

## 训练完成后评估

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_clipdiag_seed0/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_clipdiag_seed0/eval_output.txt
```

## 生成 Git 管理的实验记录

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_clipdiag_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_clipdiag_seed0/eval_output.txt \
  --output experiment_records/ppo_long_obsnorm_clipdiag_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_long_obsnorm_clipdiag_seed0.md
git commit -m "Record PPO long obs norm action clipping diagnostics seed0 summary"
git pull --rebase
git push
```

## 对比重点

和 07 对比：

- evaluation mean return 是否仍接近 `326.992`。
- tail entropy 是否仍接近 `54.67`。
- tail approx KL 是否仍偏高。
- tail PPO clip fraction 是否仍偏高。
- `action_clip_fraction` 是否很高。
- `action_clip_excess_mean` 是否持续增大。

## 本节完成标准

- `ppo_long_obsnorm_clipdiag_seed0` 完成长训。
- `experiment_records/ppo_long_obsnorm_clipdiag_seed0.md` 被 Git 管理并推回。
- 根据动作裁剪诊断决定下一节：
  - 如果动作裁剪比例很高：进入 tanh-squashed Gaussian policy。
  - 如果动作裁剪比例不高：优先检查 learning rate、update epochs 或 advantage/return normalization。
