# 07 当前任务：PPO Long Obsnorm Training

## 本节目标

把目前最强的短训配置放大成一个真正的大实验。

前面 `100k` timesteps 的 v0-v3 主要用于验证代码、诊断问题和选择配置。现在已经知道：

- observation normalization 明显有效。
- `target_kl=0.03` 太保守。
- `target_kl=0.06` 比 `0.03` 好，但仍没有超过不加 KL 的 v1。

因此本节先不继续加技巧，而是用当前最强短训配置做长训练。

## 本节实验设计

主实验：

- run name: `ppo_long_obsnorm_seed0`
- total timesteps: `3000000`
- normalize observations: `true`
- target KL: 不启用
- seed: `0`

这次实验的意义不是 smoke test，而是观察 PPO baseline 在更长训练下是否持续提升。

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
  --run-name ppo_long_obsnorm_seed0 \
  --normalize-observations
```

预计耗时取决于服务器负载。按前面 `100k` 几分钟的速度估算，`3M` 可能在数小时内完成，适合睡前运行。

## 训练完成后评估

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_seed0/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_seed0/eval_output.txt
```

这里使用 `10` 个 episode，而不是前面短训的 `5` 个 episode。长训结果更重要，需要稍微降低评估偶然性。

## 生成 Git 管理的实验记录

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_seed0/eval_output.txt \
  --output experiment_records/ppo_long_obsnorm_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_long_obsnorm_seed0.md
git commit -m "Record PPO long obs norm seed0 summary"
git pull --rebase
git push
```

## 本节需要重点观察

- evaluation mean return 是否明显超过 v1 的 `276.612`。
- tail rolling episode return 是否继续上升，还是进入平台期。
- episode length 是否明显变长。
- value loss 是否保持在可控范围。
- approx KL 和 clip fraction 是否继续偏高。
- entropy 是否明显下降，策略是否过早变得确定。

## 本节完成标准

- `ppo_long_obsnorm_seed0` 完成长训。
- `experiment_records/ppo_long_obsnorm_seed0.md` 被 Git 管理并推回。
- 根据长训结果判断下一步：
  - 如果长训显著提升：进入多 seed 稳定性验证。
  - 如果长训平台期明显：考虑 learning rate schedule、reward/return normalization 或网络结构调整。
  - 如果 KL 和 clip fraction 仍过高：重新评估 `target_kl=0.06` 或降低 update epochs。
