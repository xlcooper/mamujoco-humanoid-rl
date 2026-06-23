# 13 当前任务：PPO Squashed EP4 Multi-Seed

## 本节目标

验证 12 得到的候选 baseline 是否稳定。

当前候选配置是：

- observation normalization
- tanh-squashed Gaussian policy
- update epochs: `4`
- total timesteps: `3000000`

seed 0 的结果很好：

- evaluation mean return: `716.011`
- evaluation std: `115.490`
- action clip fraction: `0.0000`
- tail approx KL mean: `0.1038`
- tail PPO clip fraction mean: `0.4198`

但单 seed 不能证明配置可靠。本节跑 seed `1` 和 seed `2`，用于判断 Stage 1 baseline 是否可以收束。

## 本节实验设计

运行两个新 seed：

- `ppo_long_obsnorm_squash_ep4_seed1`
- `ppo_long_obsnorm_squash_ep4_seed2`

配置保持不变，只修改 `--seed` 和 `--run-name`。

## 在服务器运行 seed 1

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/train_ppo.py \
  --seed 1 \
  --total-timesteps 3000000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 4 \
  --run-name ppo_long_obsnorm_squash_ep4_seed1 \
  --normalize-observations \
  --squash-actions
```

评估：

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed1/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed1/eval_output.txt
```

生成记录：

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed1 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed1/eval_output.txt \
  --output experiment_records/ppo_long_obsnorm_squash_ep4_seed1.md
```

## 在服务器运行 seed 2

```bash
python src/train_ppo.py \
  --seed 2 \
  --total-timesteps 3000000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 4 \
  --run-name ppo_long_obsnorm_squash_ep4_seed2 \
  --normalize-observations \
  --squash-actions
```

评估：

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed2/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed2/eval_output.txt
```

生成记录：

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed2 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed2/eval_output.txt \
  --output experiment_records/ppo_long_obsnorm_squash_ep4_seed2.md
```

## 提交轻量记录

两个 seed 都完成后提交：

```bash
git add \
  experiment_records/ppo_long_obsnorm_squash_ep4_seed1.md \
  experiment_records/ppo_long_obsnorm_squash_ep4_seed2.md

git commit -m "Record PPO squashed EP4 multi-seed summaries"
git pull --rebase
git push
```

## 对比重点

汇总 seed 0/1/2：

- evaluation mean return 均值和标准差。
- evaluation episode length 是否稳定变长。
- action clip fraction 是否都为 `0`。
- tail approx KL 是否保持在合理范围。
- tail PPO clip fraction 是否低于 11 的 `0.8774`。
- 是否存在某个 seed 明显崩塌。

## 本节完成标准

- seed 1 和 seed 2 轻量实验记录被 Git 管理并推回。
- 本地汇总 seed 0/1/2。
- 判断 Stage 1 是否进入收束总结。
