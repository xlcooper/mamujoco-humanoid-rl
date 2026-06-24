# 04 当前任务：SB3 SAC Video Rendering

## 本节目标

`03` 中 SB3 SAC `1M` seed `0` evaluation mean return 达到 `6042.360`，10 个 evaluation episode 全部跑满 `1000` step。

本节目标是渲染 deterministic policy 视频，确认高 return 是否对应稳定、直观的 locomotion 行为。

本节不再重新训练，只做：

- 加载 `sac_sb3_1m_seed0` checkpoint。
- 加载 `vecnormalize.pkl`。
- 录制 deterministic evaluation 视频。
- 根据视频观察决定下一步是 SAC multi-seed，还是先写 PPO/SAC 对比总结。

## 本节代码变化

1. `src/render_sac_sb3.py`

   - 新增 SB3 SAC deterministic policy 视频渲染入口。
   - 加载 `checkpoints/sac_final.zip`。
   - 如训练启用了 observation normalization，则通过 `--vecnormalize` 加载 `vecnormalize.pkl`。
   - 使用 `render_mode="rgb_array"` 录制 mp4 视频。
   - 默认输出到 run 目录下的 `videos/`。

## 渲染命令

在 AutoDL 上运行：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl

python src/render_sac_sb3.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/checkpoints/sac_final.zip \
  --vecnormalize /root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/vecnormalize.pkl \
  --episodes 3 \
  --seed 20000 \
  --fps 30
```

默认输出目录：

```text
/root/autodl-tmp/Humanoid-runs/sac_sb3_1m_seed0/videos/
```

视频文件示例：

```text
episode_001.mp4
episode_002.mp4
episode_003.mp4
```

视频文件不提交 Git。

## 观察重点

优先确认：

- 是否能稳定站立并持续前进。
- 是否存在明显“原地抖动但拿高分”的行为。
- 是否有接近自然的交替步态。
- 身体是否长期贴地、倒地滑行或异常旋转。
- 1000 step 期间行为是否稳定，还是后半段明显退化。

## 记录方式

视频不进 Git，但观察结论要写回本 note。

如果视频显示行为合理，本节完成后应把本 note 改成“已完成总结”，并记录：

- 视频路径。
- 每个 episode 的 return / length。
- 行为观察。
- 是否支持进入 SAC multi-seed。

如果渲染报错，把完整 traceback 贴回对话；默认按 AutoDL 环境问题处理。

## 下一步候选

如果视频确认 SAC 行为质量良好：

1. 跑 SAC seed `1/2`，验证多 seed 稳定性。
2. 或先写 PPO/SAC seed `0` 对比总结，再决定是否补多 seed。

如果视频显示行为异常：

1. 保留高分但标注行为问题。
2. 优先排查 reward 组成、termination 条件和渲染行为。
3. 再决定是否调 SAC 参数或补其他诊断。
