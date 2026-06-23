# 12 当前任务：PPO Squashed Update Epochs Tuning

## 本节目标

在 tanh-squashed Gaussian policy 上降低 PPO update 强度。

11 已经解决动作越界问题：

- `action_clip_fraction`: `0.9826 -> 0.0000`
- `action_clip_excess_mean`: `19.55 -> 0.0000`

但 11 也暴露了新问题：

- evaluation mean return 从 `326.992` 降到 `283.664`
- tail `approx_kl` 从 `0.2990` 升到 `1.1188`
- tail PPO `clip_fraction` 从 `0.5571` 升到 `0.8774`

这说明 squashed policy 结构是对的，但原来的 `update_epochs=10` 对它太激进。

## 为什么先调 update epochs

PPO 每次采样一个 rollout 后，会对这批旧数据重复训练多个 epoch。

`update_epochs=10` 意味着同一批 rollout 被重复用了 10 轮。对无界 Gaussian 时已经偏猛；对 tanh-squashed policy，log_prob 还包含 tanh Jacobian 修正，策略分布更敏感，重复更新更容易让 KL 和 clip fraction 爆高。

因此本节先做最直接的 update 强度控制：

- 保留 squashed policy。
- 不启用 target KL。
- 不启用 action log std clamp。
- 将 `update_epochs` 从 `10` 降到 `4`。

## 本节实验设计

主实验：

- run name: `ppo_long_obsnorm_squash_ep4_seed0`
- total timesteps: `3000000`
- normalize observations: `true`
- squash actions: `true`
- update epochs: `4`
- target KL: 不启用
- action log std clamp: 不启用
- seed: `0`

## 在服务器运行训练

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/train_ppo.py \
  --total-timesteps 3000000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 4 \
  --run-name ppo_long_obsnorm_squash_ep4_seed0 \
  --normalize-observations \
  --squash-actions
```

## 训练完成后评估

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed0/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed0/eval_output.txt
```

## 生成 Git 管理的实验记录

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed0/eval_output.txt \
  --output experiment_records/ppo_long_obsnorm_squash_ep4_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_long_obsnorm_squash_ep4_seed0.md
git commit -m "Record PPO long obs norm squashed ep4 seed0 summary"
git pull --rebase
git push
```

## 对比重点

和 11 对比：

- evaluation mean return 是否高于 `283.664`。
- evaluation std 是否保持低于 07 的 `80.387`。
- `action_clip_fraction` 是否继续为 `0`。
- approx KL 是否明显低于 `1.1188`。
- PPO clip fraction 是否明显低于 `0.8774`。
- value loss 是否继续低于 07。
- episode length 是否比 11 更长或更稳定。

## 本节完成标准

- `ppo_long_obsnorm_squash_ep4_seed0` 完成长训。
- `experiment_records/ppo_long_obsnorm_squash_ep4_seed0.md` 被 Git 管理并推回。
- 判断 update epochs 4 是否成为 squashed policy 的候选 baseline。
