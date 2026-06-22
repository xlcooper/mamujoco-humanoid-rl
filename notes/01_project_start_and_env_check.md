# 01 当前任务：跑通 MaMuJoCo Humanoid 环境检查

## 先看这里

日常推进时优先看本文件，不用回翻聊天记录。聊天用来贴报错和讨论原因；我会把稳定结论更新回 `notes/`。

当前只做一件事：在 AutoDL 上确认 Humanoid 环境能 `reset` / `step`。不要开始 PPO 训练，也不要处理渲染录视频。

## 你现在要执行的命令

在服务器上：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
git log --oneline -3
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
```

最近提交里至少应该看到：

```text
489ea96 Fix MaMuJoCo smoke test creation
```

重新生成环境报告文件：

```bash
bash server/check_autodl_host.sh | tee server/autodl_host_report.txt
ls -lh server/autodl_host_report.txt
```

运行单智能体 smoke test：

```bash
python src/check_mamujoco_env.py --partitioning none --steps 5
```

如果通过，再运行多智能体分区 smoke test：

```bash
python src/check_mamujoco_env.py --partitioning "9|8" --steps 5
```

## 你回传给我

贴下面这些即可：

```text
1. git log --oneline -3
2. ls -lh server/autodl_host_report.txt
3. partitioning=None 的输出或报错
4. 如果 none 通过，再贴 partitioning="9|8" 的输出或报错
```

如果报错，贴 traceback 最后 80 行左右。

## 当前已确认

- 远端仓库：`git@github.com:xlcooper/mamujoco-humanoid-rl.git`
- 服务器项目目录：`/root/autodl-tmp/Humanoid`
- 独立 conda 环境：`/root/autodl-tmp/conda-envs/humanoid-rl`
- 已确认包版本：`torch 2.12.1+cu130`、`gymnasium 1.3.0`、`gymnasium_robotics 1.4.2`、`mujoco 3.9.0`、`pettingzoo 1.26.1`、`tensorboard 2.20.0`

## EGL 和 Viewer 暂时怎么处理

现在先不处理 EGL / viewer。

原因：第一阶段 PPO baseline 不需要图形窗口，也不需要录视频；只要 `render_mode=None` 的环境能正常 `reset` / `step`，训练就可以继续。你之前 Fetch 项目可能配置过 viewer、EGL 或某个 conda 环境变量，但那属于“渲染/录视频”问题，不应该挡住当前环境冒烟测试。

后面需要渲染时再查：

```bash
env | grep -E "MUJOCO|PYOPENGL|DISPLAY"
grep -R "MUJOCO_GL\|PYOPENGL_PLATFORM\|DISPLAY" ~/.bashrc ~/.profile ~/.condarc /root/autodl-tmp/conda-envs/humanoid-rl/etc/conda 2>/dev/null
```

可能会用到：

```bash
export MUJOCO_GL=egl
```

但现在不要提前改。

## 可以忽略的输出

- `AdroitHand... reward functions were updated...`：Gymnasium-Robotics 关于 Adroit 环境复现的 warning，当前 Humanoid 可以先忽略。
- `Error: unable to open display :1`：来自 `glxinfo`，说明没有图形显示窗口；不影响无渲染训练。

## 本节完成标准

完成下面两项后，进入下一节普通 PPO：

1. `python src/check_mamujoco_env.py --partitioning none --steps 5` 成功。
2. `partitioning="9|8"` 成功或有明确报错记录。
