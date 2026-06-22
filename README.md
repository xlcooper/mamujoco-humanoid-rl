# MaMuJoCo Humanoid PPO 项目

## 项目目标

本项目围绕 Farama Gymnasium-Robotics 的 MaMuJoCo Humanoid 环境，做一个可以写进算法工程师简历的强化学习项目。

阶段目标：

- Stage 1：跑通并分析普通 PPO baseline。
- Stage 2：加入工程优化和消融实验。
- Stage 3：比较单智能体 PPO 与 MaMuJoCo 多智能体分解方案。
- Stage 4：整理真实实验结论、图表和简历描述。

本项目重视三件事：算法理解、工程实现、实验纪律。所有性能结论必须来自真实 AutoDL 输出。

## 当前阶段

Stage 1：普通 PPO baseline。

- MaMuJoCo Humanoid 环境检查已完成。
- 手写 PPO 最小训练闭环已完成。
- 当前教程：`notes/03_ppo_baseline_v0.md`
- AutoDL 环境基线：`AUTODL_HOST_BASELINE.md`

## 新 AI 接手规则

编辑前先读：

1. `README.md`
2. `CHANGELOG.md`
3. `AUTODL_HOST_BASELINE.md`
4. `notes/00_project_roadmap.md`
5. 当前教程 note，例如 `notes/03_ppo_baseline_v0.md`
6. 必要时读 `autodl-project-manager/SKILL.md`

## 文档导航

- `notes/00_project_roadmap.md`：长期路线、实验阶梯、简历产出规划。
- `notes/01_project_start_and_env_check.md`：已完成，AutoDL 与 MaMuJoCo 环境检查。
- `notes/02_minimal_ppo_baseline.md`：已完成，手写 PPO 最小训练闭环和 smoke test。
- `notes/03_ppo_baseline_v0.md`：当前任务，第一条可分析 PPO baseline。
- `experiment_records/ppo_smoke_test_001.md`：已完成，PPO smoke test 轻量实验记录。
- `AUTODL_HOST_BASELINE.md`：AutoDL 硬件、CUDA、Conda 和关键包版本基线。
- `server/check_autodl_host.sh`：AutoDL 环境检查脚本。
- `scripts/summarize_ppo_run.py`：从服务器 run 目录生成 Git 管理的轻量实验记录。
- `src/`：项目代码。

## 目录约定

- `notes/`：教程式推进。每一节只承担一个阶段任务。
- `experiment_records/`：轻量实验记录，包括命令、配置摘要、指标摘要、评估结果和结论。
- `src/`：项目源码。
- `server/`：AutoDL 辅助脚本和轻量环境报告。
- `scripts/`：本地或服务器可复用工具脚本。

大型运行产物留在 AutoDL 数据盘，例如 `/root/autodl-tmp/Humanoid-runs/`，不要提交到 Git。

## 教程推进规则

1. `notes/00` 是长期路线图，其他 numbered notes 按课程顺序推进。
2. 一节完成后，把该 note 改成“已完成总结”，不要继续往里面塞下一阶段任务。
3. 下一阶段必须新建下一个 numbered note，例如 `03`、`04`。
4. 当前任务以最新 numbered note 为准，对话只用于管理、答疑和临时纠错。
5. 实验结论放进 `experiment_records/`，不要只留在聊天里。
6. 若服务器产出需要本地分析，优先生成轻量 `.md` 记录并 Git push。

## 本地与 AutoDL 分工

- 本地 Windows：编辑、Git、阅读、总结、轻量检查。
- AutoDL：依赖安装、MuJoCo 验证、PPO 训练、评估、渲染、长任务。

不要用本地机器配置推断 AutoDL 能力。

## Git 与产物规则

提交到 Git：

- 教程 notes
- 实验记录摘要
- 代码
- 小型配置和环境基线

不要提交：

- checkpoints
- TensorBoard events
- 视频
- 大量 raw logs
- 模型权重
- VNC/SSH 信息、密码、token、私钥

每次有意义更新后：

```bash
git add <files>
git commit -m "<clear message>"
git pull --rebase
git push
```

## 代码风格

- PPO 代码保持 beginner friendly。
- 使用清楚的变量名和直白控制流，避免一行塞太多逻辑。
- 在算法关键处写短注释，例如 policy forward、rollout、GAE、PPO clipping、训练主流程。
- 第一版 baseline 先保持简单，后续再加入优化技巧。

## 当前常用命令

环境 smoke test：

```bash
python src/check_mamujoco_env.py --partitioning none --steps 5
python src/check_mamujoco_env.py --partitioning "9|8" --steps 5
```

生成轻量实验记录：

```bash
python scripts/summarize_ppo_run.py --run-dir <run-dir> --eval-output <eval-output.txt> --output experiment_records/<name>.md
```

## 参考资料

- Farama MaMuJoCo Humanoid: <https://robotics.farama.org/envs/MaMuJoCo/ma_humanoid/>
- Farama MaMuJoCo overview: <https://robotics.farama.org/envs/MaMuJoCo/>
- Farama Gymnasium-Robotics installation: <https://robotics.farama.org/content/installation/>
