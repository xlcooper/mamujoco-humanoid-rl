# 05 当前任务：PPO Update Control

## 本节目标

在保留 observation normalization 的基础上，控制 PPO 每次 update 的策略变化幅度。

上一节 v1 已经证明 obs norm 有效，但也暴露出 update 过猛：

- `approx_kl` 多次达到 `0.08` 到 `0.12`
- `clip_fraction` 经常在 `0.45` 到 `0.57`

这说明 PPO 的 clip 机制频繁生效，策略更新幅度偏大。下一步要加入 KL early stopping。

## 为什么要做 KL Early Stopping

PPO 的目标是“不要让新策略离旧策略太远”。clip objective 是一种限制方式，但如果很多样本都被 clip，说明更新已经太激进。

KL early stopping 的想法很直接：

```text
每轮 PPO update 内部有多个 epoch
如果发现 new policy 和 old policy 的 KL 已经超过阈值
就提前停止这个 update
```

这样可以避免策略在一次 update 里跳太远。

## 已完成代码任务

1. 已在训练参数里增加：
   - `--target-kl`
2. 已在 PPO update 中：
   - 统计 minibatch approximate KL
   - 如果 mean KL 超过 `target_kl`，提前结束当前 update 的后续 epoch
3. 已在日志里增加：
   - 实际使用的 update epoch 数
   - 是否触发 early stop

相关代码：

- `src/ppo.py`
- `src/train_ppo.py`

## 本节实验任务

现在运行 v2：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/train_ppo.py \
  --total-timesteps 100000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 10 \
  --run-name ppo_baseline_v2_obsnorm_kl_seed0 \
  --normalize-observations \
  --target-kl 0.03
```

评估：

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_baseline_v2_obsnorm_kl_seed0/checkpoints/agent_final.pt \
  --episodes 5 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_baseline_v2_obsnorm_kl_seed0/eval_output.txt
```

生成 Git 管理的实验记录：

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_baseline_v2_obsnorm_kl_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_baseline_v2_obsnorm_kl_seed0/eval_output.txt \
  --output experiment_records/ppo_baseline_v2_obsnorm_kl_seed0.md
```

提交轻量记录：

```bash
git add experiment_records/ppo_baseline_v2_obsnorm_kl_seed0.md
git commit -m "Record PPO baseline v2 obs norm KL seed0 summary"
git pull --rebase
git push
```

## 对比重点

和 v1 对比：

- `approx_kl` 是否下降到更合理范围。
- `clip_fraction` 是否明显下降。
- evaluation mean return 是否保持或提升。
- value loss 是否仍低于 v0。
- early stop 是否频繁触发。

## 本节完成标准

- KL early stopping 代码完成并有中文注释。
- v2 实验记录通过 Git 推回。
- 根据 v1/v2 对比决定下一节方向：
  - 如果 KL 降低且性能不掉：进入更长训练。
  - 如果性能下降明显：调 target KL、learning rate 或 update epochs。
