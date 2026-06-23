# 15 当前任务：Stage 2 Final Baseline Video Rendering

## 本节目标

先暂停多智能体推进，给 Stage 2 最终单智能体 baseline 补一个展示视频。

视频用于回答两个问题：

- 分数提升后，机器人行为是否真的更稳定。
- 策略是不是存在明显怪异动作或只靠环境奖励漏洞刷分。

## 视频渲染原理

渲染不是训练，也不是回放训练数据。

脚本会加载训练好的 checkpoint，然后像 evaluation 一样运行：

```text
observation -> policy deterministic action -> environment step -> render frame
```

普通 evaluation 只记录 return 和 episode length；视频 evaluation 额外把每一步画面保存成 mp4。

## 本节代码变化

1. `src/render_policy.py`

   - 新增 checkpoint 渲染入口。
   - 复用 `evaluate.py` 的 checkpoint 加载、observation normalization 和设备选择逻辑。
   - 使用 deterministic action 录制视频，避免随机采样导致展示结果忽好忽坏。
   - 默认把 mp4 写到 `<run_dir>/videos/`。
   - 每个 episode 输出 return、length 和 video path。

2. `requirements.txt`

   - 新增 `imageio`。
   - 新增 `imageio-ffmpeg`。

## 在服务器运行

优先录 seed 1，因为它是当前三 seed 里 evaluation mean return 最高的一组。

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
pip install -r requirements.txt

MUJOCO_GL=egl python src/render_policy.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed1/checkpoints/agent_final.pt \
  --episodes 1 \
  --seed 30000 \
  --fps 30
```

如果 EGL 报错，再把 `MUJOCO_GL=egl` 去掉或改成服务器上可用的渲染后端。

## 预期输出

脚本会打印类似：

```text
episode=1 return=xxx.xxx length=xxx video=/root/autodl-tmp/Humanoid-runs/ppo_long_obsnorm_squash_ep4_seed1/videos/episode_001.mp4
```

视频文件不要提交到 Git。

## 回传轻量记录

看完视频后，新建轻量记录：

```bash
nano experiment_records/ppo_final_baseline_video_seed1.md
```

建议记录：

- checkpoint path
- video path
- episode return
- episode length
- 观察结论：是否能稳定站立/移动，是否有明显异常动作

提交：

```bash
git add experiment_records/ppo_final_baseline_video_seed1.md
git commit -m "Record PPO final baseline video summary"
git pull --rebase
git push
```

## 本节完成标准

- 至少生成 seed 1 的 1 个 mp4 视频。
- Git 中只记录轻量视频摘要，不提交 mp4。
- 本地分析后决定是否还需要 seed 0/2 各录 1 个视频。
- 完成后回到 `notes/14_stage3_multi_agent_entry.md` 继续多智能体入口检查。
