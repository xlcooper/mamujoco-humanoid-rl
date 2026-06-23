# 08 当前任务：PPO Action Std Control

## 本节目标

控制连续动作高斯策略的探索噪声。

07 的 `3M` 长训已经证明 baseline 会继续提升，但也暴露出明显问题：

- evaluation mean return: `326.992`，比 `100k` 短训更高。
- evaluation std: `80.387`，评估波动很大。
- tail entropy mean: `54.67`，远高于短训约 `24`。
- tail approx KL mean: `0.2990`，策略更新幅度过大。
- tail clip fraction mean: `0.5571`，超过一半样本被 PPO clip。

当前策略使用可学习的状态无关 `log_std`。如果它在长训中不断变大，动作会越来越随机，即使评估时用均值动作，训练过程也会变得不稳定。

## 本节代码变化

已新增：

- `PPOConfig.action_log_std_min`
- `PPOConfig.action_log_std_max`
- `ActorCritic.clamp_action_log_std(...)`
- `ActorCritic.action_log_std_metrics()`
- `train_ppo.py` 命令行参数：
  - `--action-log-std-min`
  - `--action-log-std-max`
- `metrics.csv` 新增列：
  - `action_log_std_mean`
  - `action_log_std_min`
  - `action_log_std_max`

注意：默认不传这两个参数时，旧实验行为不变。

## 本节实验设计

主实验：

- run name: `ppo_long_obsnorm_logstd05_seed0`
- total timesteps: `3000000`
- normalize observations: `true`
- target KL: 不启用
- action log std max: `0.5`
- action log std min: `-5.0`
- seed: `0`

`log_std=0.5` 对应动作标准差约 `exp(0.5)=1.65`。这不是把探索关掉，而是防止标准差长训后无限变大。

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
  --run-name ppo_long_obsnorm_logstd05_seed0 \
  --normalize-observations \
  --action-log-std-min -5.0 \
  --action-log-std-max 0.5
```

## 训练完成后评估

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_logstd05_seed0/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_logstd05_seed0/eval_output.txt
```

## 生成 Git 管理的实验记录

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_logstd05_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_logstd05_seed0/eval_output.txt \
  --output experiment_records/ppo_long_obsnorm_logstd05_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_long_obsnorm_logstd05_seed0.md
git commit -m "Record PPO long obs norm log std 05 seed0 summary"
git pull --rebase
git push
```

## 对比重点

和 `ppo_long_obsnorm_seed0` 对比：

- evaluation mean return 是否高于 `326.992`。
- evaluation std 是否低于 `80.387`。
- entropy 是否从 `54+` 明显下降。
- action log std 是否被稳定限制在 `0.5` 以内。
- approx KL 是否低于 `0.2990`。
- clip fraction 是否低于 `0.5571`。
- episode length 是否更稳定。

## 本节完成标准

- `ppo_long_obsnorm_logstd05_seed0` 完成长训。
- `experiment_records/ppo_long_obsnorm_logstd05_seed0.md` 被 Git 管理并推回。
- 判断 action std control 是否应成为后续 baseline 默认配置。
