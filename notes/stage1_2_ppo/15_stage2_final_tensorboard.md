# 15 当前任务：Stage 2 Final TensorBoard Re-Run

## 本节目标

Stage 2 的 PPO baseline 已经完成多 seed 验证。现在补一个 TensorBoard 版本的最终训练运行，用于生成可视化曲线，方便后续项目展示和复盘。

本节不是为了重新寻找更高分，而是为了补齐实验工程能力：

- 实时查看 return、loss、entropy、KL、clip fraction。
- 保存 TensorBoard event 文件用于曲线展示。
- 对照 `metrics.csv`，确认 TensorBoard 和 CSV 指标一致。

## 本节代码变化

1. `src/train_ppo.py`

   - 新增 `--tensorboard`。
   - 新增 `--tensorboard-log-dir`。
   - 训练时写入 TensorBoard event。
   - TensorBoard 指标包括 return、episode length、policy/value loss、entropy、approx KL、clip fraction、action clipping diagnostics 和 action log std。

## 为什么需要重跑

TensorBoard 不能从 checkpoint 自动还原历史训练曲线。它需要训练过程中持续写 event 文件。

之前的 Stage 2 长训保存了 `metrics.csv`，但没有写 TensorBoard event。因此如果要得到真实 TensorBoard 曲线，需要用最终配置重新跑一遍训练。

## 在服务器运行

建议先跑 seed `1`，因为它是当前 PPO final baseline 中 evaluation mean return 最高的一组。

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
pip install -r requirements.txt

python src/train_ppo.py \
  --seed 1 \
  --total-timesteps 3000000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 4 \
  --run-name ppo_final_tensorboard_seed1 \
  --normalize-observations \
  --squash-actions \
  --tensorboard
```

TensorBoard event 默认保存到：

```text
/root/autodl-tmp/Humanoid-runs/ppo_final_tensorboard_seed1/tensorboard/
```

## 实时查看 TensorBoard

训练时另开一个服务器终端：

```bash
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
tensorboard \
  --logdir /root/autodl-tmp/Humanoid-runs \
  --host 0.0.0.0 \
  --port 6006
```

然后通过 AutoDL 的端口转发或自定义服务打开 `6006`。

建议重点看：

- `charts/rolling_episode_return`
- `charts/rolling_episode_length`
- `losses/value_loss`
- `losses/entropy`
- `diagnostics/approx_kl`
- `diagnostics/clip_fraction`
- `diagnostics/action_clip_fraction`
- `policy/action_log_std_mean`

## 训练完成后评估

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_final_tensorboard_seed1/checkpoints/agent_final.pt \
  --episodes 10 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_final_tensorboard_seed1/eval_output.txt
```

## 是否提交 Git

TensorBoard event 文件不提交 Git。

如果这次重跑结果要写入项目结论，再生成轻量实验记录：

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_final_tensorboard_seed1 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_final_tensorboard_seed1/eval_output.txt \
  --output experiment_records/ppo_final_tensorboard_seed1.md
```

是否提交这个轻量记录，等你跑完后我们再看结果决定。

## 本节完成标准

- 训练时能打开 TensorBoard 并看到曲线实时更新。
- `tensorboard/` 目录下生成 event 文件。
- 完成一次 10 episodes evaluation。
- 根据结果决定是否写入 `experiment_records/`。
