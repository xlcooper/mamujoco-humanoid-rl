# 11 当前任务：PPO Tanh-Squashed Gaussian Policy

## 本节目标

解决 raw Gaussian action 大量越界的问题。

10 的诊断显示：

- tail `action_clip_fraction` 约 `0.9826`
- tail `action_clip_excess_mean` 约 `19.55`
- tail `action_log_std_mean` 约 `1.7976`
- tail `action_log_std_max` 约 `2.4999`

这说明当前无界 Gaussian policy 采样出的 raw action 几乎总是超出环境动作范围，然后被 `envs.py` 硬裁剪。PPO 训练时计算的是 raw action 的 log_prob，但环境实际执行的是 clipped action，二者不一致。

本节引入 tanh-squashed Gaussian policy，让策略天然输出环境合法动作。

## 本节代码变化

1. `src/ppo.py`
   - `ActorCritic` 增加 `squash_actions` 开关（压缩动作）
   - `ActorCritic` 保存环境动作边界对应的 `action_scale` 和 `action_bias`
   - 新增 `squash_raw_action(...)`
   - 新增 `unsquash_action(...)`
   - 新增 `squashed_log_prob(...)`
   - `get_action_and_value(...)` 支持 tanh-squashed action 采样和 log_prob 计算
   - 新增 `get_deterministic_action(...)`，评估时对 mean action 也执行 tanh 映射
2. `src/train_ppo.py`
   - 新增命令行参数 `--squash-actions`
   - 创建 `ActorCritic` 时传入环境动作上下界
   - checkpoint 新增保存 `squash_actions`、`action_low`、`action_high`
3. `src/evaluate.py`
   - 加载 checkpoint 时恢复 `squash_actions` 和动作边界
   - 评估时使用 `agent.get_deterministic_action(...)`

默认不传 `--squash-actions` 时，旧实验行为保持不变。

## 本节实验设计

主实验：

- run name: `ppo_long_obsnorm_squash_seed0`
- total timesteps: `3000000`
- normalize observations: `true`
- squash actions: `true`
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
  --update-epochs 10 \
  --run-name ppo_long_obsnorm_squash_seed0 \
  --normalize-observations \
  --squash-actions
```

## 训练完成后评估

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_seed0/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_seed0/eval_output.txt
```

## 生成 Git 管理的实验记录

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_seed0/eval_output.txt \
  --output experiment_records/ppo_long_obsnorm_squash_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_long_obsnorm_squash_seed0.md
git commit -m "Record PPO long obs norm tanh-squashed seed0 summary"
git pull --rebase
git push
```

## 对比重点

和 07/10 对比：

- evaluation mean return 是否超过 `326.992`。
- evaluation std 是否低于 `80.387`。
- `action_clip_fraction` 是否接近 `0`。
- `action_clip_excess_mean` 是否接近 `0`。
- entropy 是否低于 `54.67`。
- approx KL 是否低于 `0.2990`。
- PPO clip fraction 是否低于 `0.5571`。
- episode length 是否更稳定。

## 本节完成标准

- `ppo_long_obsnorm_squash_seed0` 完成长训。
- `experiment_records/ppo_long_obsnorm_squash_seed0.md` 被 Git 管理并推回。
- 判断 tanh-squashed Gaussian 是否成为新的 baseline 默认策略。
