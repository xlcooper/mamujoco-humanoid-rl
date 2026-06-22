# 06 当前任务：PPO KL Target Tuning

## 本节目标

调整 KL early stopping 的阈值，在“更新稳定”和“学习速度”之间找到更好的折中。

v2 说明 `target_kl=0.03` 能降低 KL 和 clip fraction，但过于保守，导致评估表现从 v1 的 `276.612` 回落到 `244.985`。

本节先不改代码，只跑一个更温和的参数版本。

## 为什么不是继续加新技巧

目前已经有：

- 手写 PPO baseline
- observation normalization
- rolling episode 日志
- KL early stopping

现在的问题不是缺新模块，而是已有 update control 的阈值需要调。直接继续加 reward scaling 或更长训练，会让变量太多，不利于判断因果。

## 本节实验任务

运行 v3：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/train_ppo.py \
  --total-timesteps 100000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 10 \
  --run-name ppo_baseline_v3_obsnorm_kl006_seed0 \
  --normalize-observations \
  --target-kl 0.06
```

评估：

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_baseline_v3_obsnorm_kl006_seed0/checkpoints/agent_final.pt \
  --episodes 5 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_baseline_v3_obsnorm_kl006_seed0/eval_output.txt
```

生成 Git 管理的实验记录：

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_baseline_v3_obsnorm_kl006_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_baseline_v3_obsnorm_kl006_seed0/eval_output.txt \
  --output experiment_records/ppo_baseline_v3_obsnorm_kl006_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_baseline_v3_obsnorm_kl006_seed0.md
git commit -m "Record PPO baseline v3 obs norm KL006 seed0 summary"
git pull --rebase
git push
```

## 对比重点

和 v1、v2 对比：

- evaluation mean return 是否接近或超过 v1 的 `276.612`。
- `approx_kl` 是否明显低于 v1，但不要像 v2 那样过度压制。
- `clip_fraction` 是否低于 v1。
- `update_epochs_used` 是否比 v2 更接近 10，说明 early stopping 不再过度频繁。
- value loss 是否保持在 v1 的较低水平附近。

## 本节完成标准

- `ppo_baseline_v3_obsnorm_kl006_seed0` 实验记录通过 Git 推回。
- 根据 v1/v2/v3 判断下一节方向：
  - 如果 v3 兼顾性能和 KL：进入更长训练。
  - 如果 v3 仍退化：考虑不使用 KL early stopping，改调 learning rate 或 update epochs。
