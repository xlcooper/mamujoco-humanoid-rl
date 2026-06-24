# 04 已完成：SB3 SAC Video Rendering

## 本节目标

`03` 中 SB3 SAC `1M` seed `0` evaluation mean return 达到 `6042.360`，10 个 evaluation episode 全部跑满 `1000` step。

本节目标是渲染 deterministic policy 视频，确认高 return 是否对应稳定、直观的 locomotion 行为。

## 本节代码变化

1. `src/render_sac_sb3.py`

   - 新增 SB3 SAC deterministic policy 视频渲染入口。
   - 加载 `checkpoints/sac_final.zip`。
   - 如训练启用了 observation normalization，则通过 `--vecnormalize` 加载 `vecnormalize.pkl`。
   - 使用 `render_mode="rgb_array"` 录制 mp4 视频。
   - 默认输出到 run 目录下的 `videos/`。

## 渲染命令

```bash
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

视频文件不提交 Git。

## 视频观察

用户已查看 `episode_003` 视频截图。

观察：

- SAC 策略能保持站立并持续移动，没有表现为倒地后滑行。
- 姿态明显前倾、屈身，步态不接近自然人类走路。
- 这种“姿态怪但高回报”的行为在 MuJoCo locomotion 中是常见现象：策略优化的是环境 reward，而不是视觉自然度或人体运动学美观度。
- 因此本项目结论应写成：SAC 在 reward 和 episode length 上显著强于当前 PPO baseline，但视频行为仍需如实描述为 reward-driven locomotion，而不是自然人形步态。

## 本节结论

视频确认了 SAC 高分策略不是训练链路错误或评估加载错误；它确实学到了能稳定存活并移动的行为。

但行为姿态不自然，说明：

- return 高不等于自然 locomotion。
- 简历和总结中应强调“连续控制 reward 表现”和“工程对比”，不要夸成“自然步态生成”。
- 多 seed 验证可以作为后续补充；当前先写 PPO/SAC 对比总结更合适。

## 下一节

进入：

- `notes/stage3_sac/05_ppo_sac_comparison_summary.md`

下一节目标：

1. 基于已有真实结果写 PPO/SAC 对比总结。
2. 明确当前对比只覆盖 SAC seed `0`，SAC multi-seed 暂缓。
3. 将视频行为观察纳入结论，避免只看 return。
