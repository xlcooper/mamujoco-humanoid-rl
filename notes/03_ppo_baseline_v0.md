# 03 当前任务：PPO Baseline v0

## 本节目标

在 `partitioning=None` 的 MaMuJoCo Humanoid 上运行第一条可分析的 PPO baseline。

这一步不是最终成绩，而是为了判断当前手写 PPO 是否有基本学习趋势，以及是否暴露稳定性问题。

## 已知基础

- 环境检查已完成：`notes/01_project_start_and_env_check.md`
- 最小训练闭环已完成：`notes/02_minimal_ppo_baseline.md`
- AutoDL 基线已记录：`AUTODL_HOST_BASELINE.md`
- Smoke test 记录：`experiment_records/ppo_smoke_test_001.md`

## 训练命令

在 AutoDL 上运行：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/train_ppo.py \
  --total-timesteps 100000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 10 \
  --run-name ppo_baseline_v0_seed0
```

## 评估命令

训练结束后评估，并保存轻量评估输出：

```bash
python src/evaluate.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_baseline_v0_seed0/checkpoints/agent_final.pt \
  --episodes 5 \
  | tee /root/autodl-tmp/Humanoid-runs/ppo_baseline_v0_seed0/eval_output.txt
```

## 生成实验记录

不要提交 run 目录、checkpoint、完整日志或视频。只生成并提交轻量实验记录：

```bash
python scripts/summarize_ppo_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/ppo_baseline_v0_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/ppo_baseline_v0_seed0/eval_output.txt \
  --output experiment_records/ppo_baseline_v0_seed0.md
```

提交：

```bash
git add experiment_records/ppo_baseline_v0_seed0.md
git commit -m "Record PPO baseline v0 seed0 summary"
git pull --rebase
git push
```

如果训练或评估报错，再把 traceback 最后 80 行贴到对话里。

## 本地分析重点

我 pull 到 `experiment_records/ppo_baseline_v0_seed0.md` 后重点看：

- episodic return 是否有上升趋势。
- episode length 是否变长。
- value loss 是否异常爆炸。
- entropy 是否过快下降。
- approx KL 和 clip fraction 是否显示更新过猛。

## 本节完成标准

- `experiment_records/ppo_baseline_v0_seed0.md` 已通过 Git 推回。
- 根据结果判断下一节方向：
  - 如果不稳定：进入 observation normalization / reward scaling。
  - 如果趋势正常：进入更长训练和多 seed。
