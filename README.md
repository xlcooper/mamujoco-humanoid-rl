# MaMuJoCo Humanoid PPO 项目

目标：围绕 Farama Gymnasium-Robotics 的 MaMuJoCo Humanoid 环境，做一个能写进算法工程师简历的强化学习项目。Windows 本机负责编辑、Git、阅读和结果分析；依赖安装、MuJoCo 验证、PPO 训练、评估和后续渲染统一在 AutoDL Linux 服务器运行。

当前项目路线：先手写普通 PPO baseline，再做可解释的稳定性优化、消融对比和 MaMuJoCo 多智能体扩展。所有实验结论必须来自真实 AutoDL 输出。

## 维护规则

1. README 只做项目导航：当前阶段、入口文件、路线和重要约定。细节写入 `notes/`、`experiment_records/`、专题文档或代码注释，避免 README 变成聊天记录。
2. 项目以教程方式推进。`notes/00` 是长期路线图；其余 numbered notes 一节对应一个阶段任务。
3. 一节完成后，必须把该 note 改成“已完成总结”，记录本节目标、完成内容、真实结果、结论和下一节入口；不要继续往旧 note 塞下一阶段任务。
4. 新阶段必须新建下一个 numbered note，例如 `notes/03_ppo_baseline_v0.md`、`notes/04_xxx.md`。当前任务永远以最新 numbered note 为准。
5. 对话只用于管理、答疑、纠错和临时决策。凡是会影响项目推进的结论，都要沉淀到 `notes/`、`experiment_records/` 或 README。
6. 实验结果使用 Git 管理轻量记录：命令、配置摘要、指标摘要、评估结果、观察和下一步判断。大型产物留在 AutoDL 数据盘。
7. 未经 AutoDL 真实运行得到的输出、指标、曲线和策略表现，不写成实验结论。
8. 用户说完成某节或某次实验后，开始修改前必须先检查 `git status --short` 和相关 note / experiment record，确认用户写入内容已读到并保留。
9. 代码风格面向初学复盘：允许适度封装，但避免一行代码包含多个逻辑；关键算法位置写短注释，例如 policy forward、rollout、GAE、PPO clipping、训练主流程。
10. 用户写在代码或 note 里的内容默认保留，只有错误、误导或明显影响后续理解时才改。
11. 每次有意义更新必须 commit、`git pull --rebase`、push。
12. 用户粘贴的运行报错默认来自 AutoDL，除非明确说明是在 Windows 本机或其他环境。

## 文档导航

| 文档 | 用途 |
| --- | --- |
| [AUTODL_HOST_BASELINE.md](AUTODL_HOST_BASELINE.md) | AutoDL 硬件、CUDA、Conda 和关键包版本基线 |
| [CHANGELOG.md](CHANGELOG.md) | 项目更新记录 |
| [notes/00_project_roadmap.md](notes/00_project_roadmap.md) | 长期路线、实验阶梯和简历产出规划 |
| [notes/01_project_start_and_env_check.md](notes/01_project_start_and_env_check.md) | 已完成：AutoDL 与 MaMuJoCo 环境检查 |
| [notes/02_minimal_ppo_baseline.md](notes/02_minimal_ppo_baseline.md) | 已完成：手写 PPO 最小训练闭环和 smoke test |
| [notes/03_ppo_baseline_v0.md](notes/03_ppo_baseline_v0.md) | 已完成：第一条可分析 PPO baseline |
| [notes/04_ppo_diagnostics_and_obs_norm.md](notes/04_ppo_diagnostics_and_obs_norm.md) | 当前任务：PPO 诊断与 observation normalization |
| [experiment_records/ppo_smoke_test_001.md](experiment_records/ppo_smoke_test_001.md) | 已完成实验：PPO smoke test 轻量记录 |
| [experiment_records/ppo_baseline_v0_seed0.md](experiment_records/ppo_baseline_v0_seed0.md) | 已完成实验：PPO baseline v0 |
| [scripts/summarize_ppo_run.py](scripts/summarize_ppo_run.py) | 从服务器 run 目录生成 Git 管理的轻量实验记录 |
| [server/check_autodl_host.sh](server/check_autodl_host.sh) | AutoDL 环境检查脚本 |
| [src/](src/) | PPO、环境适配和评估代码 |

## 当前阶段

当前阶段：Stage 1，普通 PPO baseline。

已完成：

- MaMuJoCo Humanoid `partitioning=None` 和 `partitioning="9|8"` 环境检查。
- AutoDL 环境基线整理。
- 手写 PPO 最小训练闭环。
- PPO smoke test 轻量实验记录。
- PPO baseline v0 实验记录和分析。
- Observation normalization 与 rolling episode 日志代码。

当前教程：

```text
notes/04_ppo_diagnostics_and_obs_norm.md
```

新对话接手时，先阅读：

```text
README.md
CHANGELOG.md
AUTODL_HOST_BASELINE.md
notes/00_project_roadmap.md
notes/04_ppo_diagnostics_and_obs_norm.md
experiment_records/ppo_smoke_test_001.md
experiment_records/ppo_baseline_v0_seed0.md
```

## 目录约定

| 路径 | 内容 |
| --- | --- |
| `notes/` | 教程笔记、阶段总结、当前任务 |
| `experiment_records/` | Git 管理的轻量实验记录：命令、配置、指标摘要、评估结果、结论 |
| `src/` | 项目源码：环境适配、PPO、训练、评估 |
| `scripts/` | 本地或服务器可复用工具脚本 |
| `server/` | AutoDL 环境检查和服务器辅助脚本 |
| `/root/autodl-tmp/Humanoid-runs/` | AutoDL 运行产物：config、metrics、checkpoints、评估输出等 |

训练和评估采用两层输出：

```text
/root/autodl-tmp/Humanoid-runs/
```

保存完整运行产物，包括 checkpoint、完整 `metrics.csv`、评估输出、未来可能的视频或 TensorBoard event。这些文件可能很大，不提交到 Git。

```text
experiment_records/
```

保存可提交到 Git 的轻量记录，用于本地分析、阶段复盘和简历材料整理。

不得提交：VNC 密码、SSH 地址和端口、私钥、token、训练生成的大文件、服务器当前运行状态、checkpoint、视频和 TensorBoard event。

## 教程路线

| 节次 | 主题 | 状态 |
| --- | --- | --- |
| 00 | 项目路线图 | 已建立 |
| 01 | AutoDL 与 MaMuJoCo 环境检查 | 已完成 |
| 02 | 手写 PPO 最小训练闭环 | 已完成 |
| 03 | PPO Baseline v0 | 已完成 |
| 04 | PPO 诊断与 observation normalization | 当前进行中，等待服务器运行 v1 |

## 当前任务入口

按 `notes/04_ppo_diagnostics_and_obs_norm.md` 先完成代码改进，再在 AutoDL 上运行 baseline v1，并生成：

```text
experiment_records/ppo_baseline_v1_obsnorm_seed0.md
```

实验记录生成命令模板：

```bash
python scripts/summarize_ppo_run.py \
  --run-dir <run-dir> \
  --eval-output <eval-output.txt> \
  --output experiment_records/<name>.md
```

生成后只提交轻量记录：

```bash
git add experiment_records/<name>.md
git commit -m "Record <experiment name> summary"
git pull --rebase
git push
```

## 官方资料入口

- [Farama MaMuJoCo Humanoid](https://robotics.farama.org/envs/MaMuJoCo/ma_humanoid/)
- [Farama MaMuJoCo overview](https://robotics.farama.org/envs/MaMuJoCo/)
- [Farama Gymnasium-Robotics installation](https://robotics.farama.org/content/installation/)
