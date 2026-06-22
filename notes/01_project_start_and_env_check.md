# 01 当前任务：服务器仓库、独立环境与 MaMuJoCo 冒烟测试

## 当前状态

- 本地仓库已经连接远端：`git@github.com:xlcooper/mamujoco-humanoid-rl.git`
- 项目方向确定为 MaMuJoCo Humanoid RL，第一阶段先跑通普通 PPO。
- 你服务器上以前跑过 Fetch 项目，可能已有 Gymnasium-Robotics / MuJoCo 相关 conda 环境。
- 这次 Humanoid 项目建议使用独立环境，避免污染 Fetch 项目，也方便以后写复现实验说明。
- 旧 Fetch 项目之前误放在系统盘，如果确认已完整推送到远端，可以迁移或删除，避免系统盘被训练产物占满。
- 已确认的服务器包版本：`torch 2.12.1+cu130`、`gymnasium 1.3.0`、`gymnasium_robotics 1.4.2`、`mujoco 3.9.0`、`pettingzoo 1.26.1`、`tensorboard 2.20.0`。
- 当前已知问题：第一次提交环境报告时没有生成 `server/autodl_host_report.txt` 文件；第一次 smoke test 使用的是旧脚本，还没有拉到 `489ea96 Fix MaMuJoCo smoke test creation`。

这一节的目标不是训练 PPO，而是把服务器工作区和 Python 环境整理干净，并确认 MaMuJoCo Humanoid 能正常 `reset` / `step`。

## 本节完成标准

完成本节后，我们才能进入下一节“普通 PPO 最小训练闭环”：

1. 服务器上的本项目位于数据盘，例如 `/root/autodl-tmp/Humanoid`。
2. 独立 Python 环境已创建，不复用 Fetch 项目的训练环境。
3. `server/check_autodl_host.sh` 已运行，关键环境信息已记录。
4. `partitioning=None` 的 Humanoid smoke test 通过。
5. `partitioning="9|8"` 的 Humanoid smoke test 有明确结果。
6. 你把关键输出或报错贴回来。

## Step 1：把本项目放到数据盘

在 AutoDL 上先进入数据盘：

```bash
cd /root/autodl-tmp
```

如果这里还没有本项目，就 clone：

```bash
git clone git@github.com:xlcooper/mamujoco-humanoid-rl.git Humanoid
cd Humanoid
```

如果服务器已经有这个仓库，但放在系统盘，建议迁移到数据盘。先确认旧位置没有未提交内容：

```bash
cd /旧的/Humanoid/路径
git status --short
git remote -v
git log --oneline -3
```

如果 `git status --short` 为空，并且远端正确，可以直接在数据盘重新 clone。旧目录先不要马上硬删，建议先改名备份：

```bash
mv /旧的/Humanoid/路径 /root/autodl-tmp/Humanoid_old_backup
```

确认新目录能正常运行后，再删除备份。

## Step 2：不要直接改 Fetch 的旧环境

旧 Fetch 环境可以用来查看版本，但不要在里面直接 `pip install -r requirements.txt`。原因是这会修改旧项目的依赖，之后如果 Fetch 项目要复现，环境可能已经变了。

可以先查看旧环境供参考：

```bash
conda env list
conda activate 你的fetch环境名

python - <<'PY'
import importlib.metadata as md

for package in ["torch", "gymnasium", "gymnasium-robotics", "mujoco", "pettingzoo"]:
    try:
        print(package, md.version(package))
    except Exception:
        print(package, "missing")
PY
```

看完后退出即可：

```bash
conda deactivate
```

## Step 3：创建 Humanoid 独立环境

推荐把环境也放在数据盘，减少系统盘压力：

```bash
mkdir -p /root/autodl-tmp/conda-envs
conda create -p /root/autodl-tmp/conda-envs/humanoid-rl python=3.11 -y
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
```

确认你在项目目录：

```bash
cd /root/autodl-tmp/Humanoid
```

安装依赖：

```bash
pip install -r requirements.txt
```

先不锁死版本。等本节跑通后，我们根据真实 AutoDL 输出决定是否写 `requirements-lock.txt` 或环境说明。

## Step 4：采集服务器环境报告

在项目根目录执行：

```bash
bash server/check_autodl_host.sh | tee server/autodl_host_report.txt
```

注意：只运行 `bash server/check_autodl_host.sh` 会把报告打印到终端，但不会保存成文件。运行后先确认文件存在：

```bash
ls -lh server/autodl_host_report.txt
tail -n 40 server/autodl_host_report.txt
```

这份报告主要用于后续写 `AUTODL_HOST_BASELINE.md`。报告里不要加入 SSH、VNC 密码、token 或任何私密信息。

如果你想直接从服务器提交报告：

```bash
git add server/autodl_host_report.txt
git commit -m "Record AutoDL host report"
git pull --rebase
git push
```

如果提交不方便，就把关键输出贴回来。

## Step 5：运行 MaMuJoCo 冒烟测试

先确认服务器已经拉到最新脚本：

```bash
git pull --rebase
git log --oneline -3
```

最近提交里应该能看到：

```text
489ea96 Fix MaMuJoCo smoke test creation
```

先测单智能体版本，这是普通 PPO baseline 会用的目标：

```bash
python src/check_mamujoco_env.py --partitioning none --steps 5
```

再测 `9|8` 分区版本，为后续多智能体扩展做准备：

```bash
python src/check_mamujoco_env.py --partitioning "9|8" --steps 5
```

预期只看是否能跑通：

- 能打印 `possible_agents` 和 `active_agents`
- 能打印 observation/action space
- 随机动作能连续 step
- reward 不需要好看，因为当前不是训练

已知非阻塞信息：

- `AdroitHand... reward functions were updated...` 是 Gymnasium-Robotics 关于 Adroit 环境版本复现的 warning，当前 Humanoid smoke test 可以先忽略。
- `Error: unable to open display :1` 来自 `glxinfo`，说明当前没有可用显示窗口；无渲染训练和 headless smoke test 不依赖它。后续如果要录视频或渲染，再单独处理 `MUJOCO_GL=egl`。

## Step 6：旧 Fetch 项目怎么处理

如果旧 Fetch 项目在系统盘，处理顺序建议是：

1. 进入旧项目目录。
2. 跑 `git status --short`，确认没有未提交修改。
3. 跑 `git remote -v`，确认远端正确。
4. 跑 `git log --oneline -3`，确认最近提交没问题。
5. 如果还不放心，先移动到数据盘备份。
6. Humanoid 项目跑通后，再删除旧备份。

不要删除旧 conda 环境，除非你确认 Fetch 项目以后完全不需要复现。当前只建议先不使用它。

## 你需要回传给我

请把下面这些结果贴回来：

```text
1. AutoDL GPU / CUDA / Python / conda 环境路径
2. torch, gymnasium, gymnasium-robotics, mujoco, pettingzoo 的版本
3. partitioning=None smoke test 是否通过
4. partitioning="9|8" smoke test 是否通过
5. 如果报错，贴 traceback 最后 80 行左右
```

拿到这些信息后，我会把稳定事实整理进 `AUTODL_HOST_BASELINE.md`，然后开始下一节：普通 PPO 的最小可运行训练代码。
