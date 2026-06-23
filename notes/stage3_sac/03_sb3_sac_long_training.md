# 03 当前任务：SB3 SAC Long Training

## 本节目标

`02` 已经确认 SB3 SAC smoke test 链路可用。本节开始跑第一条可分析的 SAC baseline。

本节仍然先做 seed `0`，目标是判断 SB3 SAC 在同一个 `partitioning=None` Humanoid 环境中是否具备比短训更强的学习趋势，并为后续 `1M-3M` timesteps、多 seed 和 PPO/SAC 对比打基础。

本节不是最终结论，重点是：

- 跑 `1M` timesteps 的 SB3 SAC seed `0`。
- 保存 TensorBoard、Monitor、checkpoint、VecNormalize 和 evaluation 输出。
- 生成 Git 管理的轻量实验记录。
- 初步对比 PPO final baseline。

PPO final baseline 对照：

- observation normalization
- tanh-squashed Gaussian policy
- update epochs: `4`
- total timesteps: `3000000`
- seed 0 evaluation mean return: `716.011`
- seed 1 evaluation mean return: `899.806`
- seed 2 evaluation mean return: `861.418`
- three-seed mean over evaluation means: `825.745`

## 训练命令

在 AutoDL 上运行：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
pip install -r requirements.txt

python src/train_sac_sb3.py \
  --seed 0 \
  --total-timesteps 1000000 \
  --learning-starts 10000 \
  --batch-size 256 \
  --buffer-size 1000000 \
  --run-name sac_sb3_1m_seed0 \
  --eval-episodes 10 \
  --log-interval 10
```

默认使用：

- `ent_coef="auto"`
- `target_entropy="auto"`
- `train_freq=1`
- `gradient_steps=1`
- `gamma=0.99`
- `tau=0.005`
- `VecNormalize(norm_obs=True, norm_reward=False)`
- `MlpPolicy` with `net_arch=[256, 256]`

## 实时查看 TensorBoard

另开一个服务器终端：

```bash
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
tensorboard \
  --logdir /root/autodl-tmp/Humanoid-runs \
  --host 0.0.0.0 \
  --port 6006
```

优先看：

- `rollout/ep_rew_mean`
- `rollout/ep_len_mean`
- `train/actor_loss`
- `train/critic_loss`
- `train/ent_coef`
- `train/ent_coef_loss`
- `time/fps`

## 训练结束后评估

训练命令结束时会自动评估一次。为了和 PPO 阶段一致，建议训练结束后单独复评并保存输出：

```bash
python src/evaluate_sac_sb3.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/checkpoints/sac_final.zip \
  --vecnormalize /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/vecnormalize.pkl \
  --episodes 10 \
  --output-json /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/eval_results.json \
  | tee /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/eval_output.txt
```

## 生成 Git 管理的轻量记录

```bash
python scripts/summarize_sac_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/eval_output.txt \
  --eval-json /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/eval_results.json \
  --output experiment_records/sac_sb3_1m_seed0.md
```

只提交轻量记录，不提交 run 目录、checkpoint、TensorBoard event 或 replay buffer：

```bash
git add experiment_records/sac_sb3_1m_seed0.md
git commit -m "Record SB3 SAC 1M seed0 summary"
git pull --rebase
git push
```

如果训练或评估报错，把完整 traceback 贴回对话；如果没报错，就通过 Git 管理结果。我本地 pull 后再分析。

## 输出路径

默认 run 目录：

```text
/root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/
```

需要确认的关键产物：

```text
config.json
monitor.monitor.csv
tensorboard/
checkpoints/sac_final.zip
vecnormalize.pkl
eval_results.json
eval_output.txt
```

## 分析重点

和 smoke test 对比：

- `rollout/ep_rew_mean` 是否持续超过 `186`。
- `rollout/ep_len_mean` 是否持续超过 `40.4`。
- evaluation mean return 是否明显超过 smoke test 的 `207.056`。
- episode length 是否从 40 step 左右进入更长的稳定区间。

和 PPO final baseline 对比：

- SAC seed `0` 的 evaluation mean return 与 PPO seed `0` 的 `716.011` 对比。
- SAC seed `0` 的 episode length 与 PPO final 视频/评估表现对比。
- TensorBoard 中 SAC critic loss、actor loss、entropy coefficient 是否稳定。

## 本节通过标准

- `1M` timesteps 训练完成。
- run 目录生成 checkpoint、VecNormalize、TensorBoard、Monitor 和 evaluation 输出。
- 生成并提交 `experiment_records/sac_sb3_1m_seed0.md`。
- 能基于真实结果判断下一步是：
  - 继续 `3M` timesteps 长训；
  - 做 SAC 参数调整；
  - 先补视频渲染确认行为质量；
  - 或进入多 seed 验证。
