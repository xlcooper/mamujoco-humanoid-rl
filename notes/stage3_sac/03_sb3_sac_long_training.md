# 03 已完成：SB3 SAC Long Training

## 本节目标

`02` 已经确认 SB3 SAC smoke test 链路可用。本节运行第一条可分析的 SAC baseline：

- 跑 `1M` timesteps 的 SB3 SAC seed `0`。
- 保存 TensorBoard、Monitor、checkpoint、VecNormalize 和 evaluation 输出。
- 生成 Git 管理的轻量实验记录。
- 初步对比 PPO final baseline。

## 训练配置

实验记录：

- `experiment_records/sac_sb3_1m_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `1000000`
- learning starts: `10000`
- replay buffer size: `1000000`
- batch size: `256`
- train frequency: `1`
- gradient steps: `1`
- entropy coefficient: `auto`
- target entropy: `auto`
- observation normalization: SB3 `VecNormalize(norm_obs=True, norm_reward=False)`
- network: `MlpPolicy`, `net_arch=[256, 256]`

训练命令：

```bash
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

## 真实结果

deterministic evaluation：

```text
episode=1 return=6086.414 length=1000
episode=2 return=6043.676 length=1000
episode=3 return=6016.683 length=1000
episode=4 return=6070.044 length=1000
episode=5 return=6017.678 length=1000
episode=6 return=6075.439 length=1000
episode=7 return=6035.541 length=1000
episode=8 return=6097.653 length=1000
episode=9 return=5977.473 length=1000
episode=10 return=6003.000 length=1000
mean_return=6042.360 std_return=37.329
mean_length=1000.000
```

Monitor tail 观察：

- tail episode return 多数在 `5800-6100` 区间。
- tail episode length 绝大多数达到 `1000` step 时间上限。
- tail 中出现一次 `581` step episode，return 为 `3464.562`，但整体评估结果稳定。

## 与 PPO Final Baseline 对比

PPO final baseline：

- seed 0 evaluation mean return: `716.011`
- seed 1 evaluation mean return: `899.806`
- seed 2 evaluation mean return: `861.418`
- three-seed mean over evaluation means: `825.745`

SAC `1M` seed `0`：

- evaluation mean return: `6042.360`
- evaluation std: `37.329`
- mean episode length: `1000.000`

初步判断：

- 在当前 seed `0` 上，SB3 SAC 明显强于 Stage 2 PPO final baseline。
- SAC 只训练 `1M` timesteps 就达到全部 evaluation episode 跑满 `1000` step，说明 off-policy SAC 在该 Humanoid 单智能体任务上非常有效。
- 但目前仍只有 seed `0`，还不能直接写成多 seed 稳定性结论。

## 本节结论

Stage 3 第一条 SAC baseline 成功。

这说明：

- SB3 SAC 不只是工程链路跑通，而是在 `1M` timesteps 内学到了高回报策略。
- 当前最重要的下一步不是继续盲目加到 `3M`，而是先渲染视频确认高 return 对应稳定 locomotion 行为。
- 视频确认后，再决定做 SAC seed `1/2` 多 seed 验证，或整理 PPO/SAC 对比总结。

## 下一节

进入：

- `notes/stage3_sac/04_sb3_sac_video_rendering.md`

下一节目标：

1. 使用 `src/render_sac_sb3.py` 渲染 SAC seed `0` deterministic evaluation 视频。
2. 确认 `6042.360` mean return 对应的行为是否是稳定 locomotion。
3. 根据视频表现决定下一步是 SAC multi-seed，还是先写 PPO/SAC 对比阶段总结。
