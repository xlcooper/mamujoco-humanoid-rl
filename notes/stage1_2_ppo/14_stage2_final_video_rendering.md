# 14 已完成/可复用：Stage 2 Final Baseline Video Rendering

## 本节目标

给 Stage 2 最终单智能体 baseline 补一个展示视频。

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

## 是否提交 Git

视频不是训练结果，也不是数值评估结果。视频文件不提交 Git，视频路径和主观观察也不强制写轻量实验记录。

如果后续做项目展示，只需要保留 mp4 文件或从服务器下载到本地展示目录。

## 本节完成标准

- 至少生成 seed 1 的 1 个 mp4 视频。
- 确认 `MUJOCO_GL=egl` 可以在当前 `humanoid-rl` 环境中完成 headless rendering。
- 本地观看视频，校准对 PPO final baseline 的行为描述。
- 完成后进入 `notes/stage1_2_ppo/15_stage2_final_tensorboard.md`，补齐 TensorBoard 曲线。
