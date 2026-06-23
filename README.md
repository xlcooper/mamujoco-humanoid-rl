# MaMuJoCo Humanoid PPO 项目

目标：围绕 Farama Gymnasium-Robotics 的 MaMuJoCo Humanoid 环境，做一个能写进算法工程师简历的强化学习项目。Windows 本机负责编辑、Git、阅读和结果分析；依赖安装、MuJoCo 验证、PPO 训练、评估和后续渲染统一在 AutoDL Linux 服务器运行。

当前项目路线：先手写普通 PPO baseline，再做可解释的稳定性优化和消融对比；PPO 收束后引入 SAC 作为 off-policy 强基线，再视进度扩展到 MaMuJoCo 多智能体。所有实验结论必须来自真实 AutoDL 输出。

## 维护规则

1. README 只做项目导航：当前阶段、入口文件、路线和重要约定。细节写入 `notes/`、`experiment_records/`、专题文档或代码注释，避免 README 变成聊天记录。
2. 项目以教程方式推进。`notes/00_project_roadmap.md` 是长期路线图；其余 numbered notes 按 stage 子目录组织，一节对应一个阶段任务。
3. 一节完成后，必须把该 note 改成“已完成总结”，记录本节目标、完成内容、真实结果、结论和下一节入口；不要继续往旧 note 塞下一阶段任务。
4. 新阶段必须放入对应 stage 子目录并新建 numbered note，例如 `notes/stage1_2_ppo/03_ppo_baseline_v0.md`、`notes/stage3_sac/01_stage3_sac_entry.md`。`notes/00_project_roadmap.md` 留在根目录作为总路线图。
5. 对话只用于管理、答疑、纠错和临时决策。凡是会影响项目推进的结论，都要沉淀到 `notes/`、`experiment_records/` 或 README。
6. 实验结果使用 Git 管理轻量记录：命令、配置摘要、指标摘要、评估结果、观察和下一步判断。大型产物留在 AutoDL 数据盘。
7. 未经 AutoDL 真实运行得到的输出、指标、曲线和策略表现，不写成实验结论。
8. 用户说完成某节或某次实验后，开始修改前必须先检查 `git status --short` 和相关 note / experiment record，确认用户写入内容已读到并保留。
9. 代码风格面向初学复盘：允许适度封装，但避免一行代码包含多个逻辑；关键算法位置写短注释，例如 policy forward、rollout、GAE、PPO clipping、训练主流程。
10. 用户写在代码或 note 里的内容默认保留，只有错误、误导或明显影响后续理解时才改。
11. note 中如有“已完成代码”或“本节代码变化”，必须按文件列出：文件名、具体新增/修改的函数/参数/日志列、行为变化；不能只列属性名或概念名。
12. 用户说“把规则记一下”时，必须把规则写入 README 或对应约束文档；这句话本身也是项目规则。
13. 每次有意义更新必须 commit、`git pull --rebase`、push。
14. 用户粘贴的运行报错默认来自 AutoDL，除非明确说明是在 Windows 本机或其他环境。

## 文档导航

| 文档 | 用途 |
| --- | --- |
| [AUTODL_HOST_BASELINE.md](AUTODL_HOST_BASELINE.md) | AutoDL 硬件、CUDA、Conda 和关键包版本基线 |
| [CHANGELOG.md](CHANGELOG.md) | 项目更新记录 |
| [notes/00_project_roadmap.md](notes/00_project_roadmap.md) | 长期路线、实验阶梯和简历产出规划 |
| [notes/stage0_setup/01_project_start_and_env_check.md](notes/stage0_setup/01_project_start_and_env_check.md) | 已完成：AutoDL 与 MaMuJoCo 环境检查 |
| [notes/stage1_2_ppo/02_minimal_ppo_baseline.md](notes/stage1_2_ppo/02_minimal_ppo_baseline.md) | 已完成：手写 PPO 最小训练闭环和 smoke test |
| [notes/stage1_2_ppo/03_ppo_baseline_v0.md](notes/stage1_2_ppo/03_ppo_baseline_v0.md) | 已完成：第一条可分析 PPO baseline |
| [notes/stage1_2_ppo/04_ppo_diagnostics_and_obs_norm.md](notes/stage1_2_ppo/04_ppo_diagnostics_and_obs_norm.md) | 已完成：PPO 诊断与 observation normalization |
| [notes/stage1_2_ppo/05_ppo_update_control.md](notes/stage1_2_ppo/05_ppo_update_control.md) | 已完成：PPO update control |
| [notes/stage1_2_ppo/06_ppo_kl_target_tuning.md](notes/stage1_2_ppo/06_ppo_kl_target_tuning.md) | 已完成：PPO KL target tuning |
| [notes/stage1_2_ppo/07_ppo_long_obsnorm_training.md](notes/stage1_2_ppo/07_ppo_long_obsnorm_training.md) | 已完成：PPO long obs norm training |
| [notes/stage1_2_ppo/08_ppo_action_std_control.md](notes/stage1_2_ppo/08_ppo_action_std_control.md) | 已完成：PPO action std control |
| [notes/stage1_2_ppo/09_ppo_relaxed_action_std_control.md](notes/stage1_2_ppo/09_ppo_relaxed_action_std_control.md) | 已完成：PPO relaxed action std control |
| [notes/stage1_2_ppo/10_ppo_action_clipping_diagnostics.md](notes/stage1_2_ppo/10_ppo_action_clipping_diagnostics.md) | 已完成：PPO action clipping diagnostics |
| [notes/stage1_2_ppo/11_ppo_tanh_squashed_policy.md](notes/stage1_2_ppo/11_ppo_tanh_squashed_policy.md) | 已完成：PPO tanh-squashed Gaussian policy |
| [notes/stage1_2_ppo/12_ppo_squashed_update_epochs_tuning.md](notes/stage1_2_ppo/12_ppo_squashed_update_epochs_tuning.md) | 已完成：PPO squashed update epochs tuning |
| [notes/stage1_2_ppo/13_ppo_squashed_ep4_multiseed.md](notes/stage1_2_ppo/13_ppo_squashed_ep4_multiseed.md) | 已完成：PPO squashed EP4 multi-seed |
| [notes/stage1_2_ppo/14_stage2_final_video_rendering.md](notes/stage1_2_ppo/14_stage2_final_video_rendering.md) | 已完成/可复用：Stage 2 final baseline video rendering |
| [notes/stage1_2_ppo/15_stage2_final_tensorboard.md](notes/stage1_2_ppo/15_stage2_final_tensorboard.md) | 暂缓任务：Stage 2 final TensorBoard re-run |
| [notes/stage3_sac/01_stage3_sac_entry.md](notes/stage3_sac/01_stage3_sac_entry.md) | 已完成规划：Stage 3 SB3 SAC baseline and off-policy comparison |
| [notes/stage3_sac/02_sb3_sac_smoke_test.md](notes/stage3_sac/02_sb3_sac_smoke_test.md) | 当前任务：SB3 SAC smoke test |
| [experiment_records/ppo_smoke_test_001.md](experiment_records/ppo_smoke_test_001.md) | 已完成实验：PPO smoke test 轻量记录 |
| [experiment_records/ppo_baseline_v0_seed0.md](experiment_records/ppo_baseline_v0_seed0.md) | 已完成实验：PPO baseline v0 |
| [experiment_records/ppo_baseline_v1_obsnorm_seed0.md](experiment_records/ppo_baseline_v1_obsnorm_seed0.md) | 已完成实验：PPO baseline v1 obs norm |
| [experiment_records/ppo_baseline_v2_obsnorm_kl_seed0.md](experiment_records/ppo_baseline_v2_obsnorm_kl_seed0.md) | 已完成实验：PPO baseline v2 obs norm + KL |
| [experiment_records/ppo_baseline_v3_obsnorm_kl006_seed0.md](experiment_records/ppo_baseline_v3_obsnorm_kl006_seed0.md) | 已完成实验：PPO baseline v3 obs norm + KL 0.06 |
| [experiment_records/ppo_long_obsnorm_seed0.md](experiment_records/ppo_long_obsnorm_seed0.md) | 已完成实验：PPO long obs norm |
| [experiment_records/ppo_long_obsnorm_logstd05_seed0.md](experiment_records/ppo_long_obsnorm_logstd05_seed0.md) | 已完成实验：PPO long obs norm + log std max 0.5 |
| [experiment_records/ppo_long_obsnorm_logstd10_seed0.md](experiment_records/ppo_long_obsnorm_logstd10_seed0.md) | 已完成实验：PPO long obs norm + log std max 1.0 |
| [experiment_records/ppo_long_obsnorm_clipdiag_seed0.md](experiment_records/ppo_long_obsnorm_clipdiag_seed0.md) | 已完成实验：PPO action clipping diagnostics |
| [experiment_records/ppo_long_obsnorm_squash_seed0.md](experiment_records/ppo_long_obsnorm_squash_seed0.md) | 已完成实验：PPO tanh-squashed Gaussian policy |
| [experiment_records/ppo_long_obsnorm_squash_ep4_seed0.md](experiment_records/ppo_long_obsnorm_squash_ep4_seed0.md) | 已完成实验：PPO squashed EP4 seed 0 |
| [experiment_records/ppo_long_obsnorm_squash_ep4_seed1.md](experiment_records/ppo_long_obsnorm_squash_ep4_seed1.md) | 已完成实验：PPO squashed EP4 seed 1 |
| [experiment_records/ppo_long_obsnorm_squash_ep4_seed2.md](experiment_records/ppo_long_obsnorm_squash_ep4_seed2.md) | 已完成实验：PPO squashed EP4 seed 2 |
| [scripts/summarize_ppo_run.py](scripts/summarize_ppo_run.py) | 从服务器 run 目录生成 Git 管理的轻量实验记录 |
| [scripts/summarize_sac_run.py](scripts/summarize_sac_run.py) | 从服务器 SB3 SAC run 目录生成 Git 管理的轻量实验记录 |
| [src/render_policy.py](src/render_policy.py) | 加载 checkpoint 并录制 deterministic evaluation 视频 |
| [src/evaluate_sac_sb3.py](src/evaluate_sac_sb3.py) | 加载 SB3 SAC checkpoint 并执行 deterministic evaluation |
| [server/check_autodl_host.sh](server/check_autodl_host.sh) | AutoDL 环境检查脚本 |
| [src/](src/) | PPO、环境适配和评估代码 |

## 当前阶段

当前阶段：Stage 3 SB3 SAC smoke test。Stage 2 final TensorBoard re-run 暂缓，它只补充展示曲线，不阻塞 Stage 3 的 SAC 强基线推进。

已完成：

- MaMuJoCo Humanoid `partitioning=None` 和 `partitioning="9|8"` 环境检查。
- AutoDL 环境基线整理。
- 手写 PPO 最小训练闭环。
- PPO smoke test 轻量实验记录。
- PPO baseline v0 实验记录和分析。
- Observation normalization 与 rolling episode 日志代码。
- PPO baseline v1 obs norm 实验记录和分析。
- PPO target KL early stopping 代码。
- PPO baseline v2 obs norm + KL 实验记录和分析。
- PPO baseline v3 obs norm + KL 0.06 实验记录和分析。
- PPO long obs norm 实验记录和分析。
- PPO action log std clamp 与日志诊断代码。
- PPO long obs norm + log std max 0.5 实验记录和分析。
- PPO long obs norm + log std max 1.0 实验记录和分析。
- PPO action clipping diagnostics 代码。
- PPO action clipping diagnostics 实验记录和分析。
- PPO tanh-squashed Gaussian policy 代码。
- PPO tanh-squashed Gaussian policy 实验记录和分析。
- PPO squashed update epochs tuning 实验记录和分析。
- PPO squashed EP4 seed `0/1/2` 多 seed 稳定性验证。
- Stage 3 SAC 路线规划，明确使用 SB3 SAC，不再手写 SAC。

最终单智能体 baseline：

- observation normalization
- tanh-squashed Gaussian policy
- update epochs: `4`
- total timesteps: `3000000`
- seed 0 evaluation mean return: `716.011`
- seed 1 evaluation mean return: `899.806`
- seed 2 evaluation mean return: `861.418`
- three-seed mean over evaluation means: `825.745`

当前教程：

```text
notes/stage3_sac/02_sb3_sac_smoke_test.md
```

新对话接手时，先阅读：

```text
README.md
CHANGELOG.md
AUTODL_HOST_BASELINE.md
notes/00_project_roadmap.md
notes/stage1_2_ppo/13_ppo_squashed_ep4_multiseed.md
notes/stage1_2_ppo/15_stage2_final_tensorboard.md
notes/stage3_sac/01_stage3_sac_entry.md
notes/stage3_sac/02_sb3_sac_smoke_test.md
experiment_records/ppo_smoke_test_001.md
experiment_records/ppo_baseline_v0_seed0.md
experiment_records/ppo_baseline_v1_obsnorm_seed0.md
experiment_records/ppo_baseline_v2_obsnorm_kl_seed0.md
experiment_records/ppo_baseline_v3_obsnorm_kl006_seed0.md
experiment_records/ppo_long_obsnorm_seed0.md
experiment_records/ppo_long_obsnorm_logstd05_seed0.md
experiment_records/ppo_long_obsnorm_logstd10_seed0.md
experiment_records/ppo_long_obsnorm_clipdiag_seed0.md
experiment_records/ppo_long_obsnorm_squash_seed0.md
experiment_records/ppo_long_obsnorm_squash_ep4_seed0.md
experiment_records/ppo_long_obsnorm_squash_ep4_seed1.md
experiment_records/ppo_long_obsnorm_squash_ep4_seed2.md
```

## 目录约定

| 路径 | 内容 |
| --- | --- |
| `notes/` | 教程笔记、阶段总结、当前任务 |
| `notes/stage0_setup/` | Stage 0 环境与项目启动 |
| `notes/stage1_2_ppo/` | Stage 1-2 手写 PPO baseline、优化、最终视频与 TensorBoard |
| `notes/stage3_sac/` | Stage 3 SB3 SAC baseline 与 PPO/SAC 对比 |
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
| 04 | PPO 诊断与 observation normalization | 已完成 |
| 05 | PPO update control | 已完成 |
| 06 | PPO KL target tuning | 已完成 |
| 07 | PPO long obs norm training | 已完成 |
| 08 | PPO action std control | 已完成 |
| 09 | PPO relaxed action std control | 已完成 |
| 10 | PPO action clipping diagnostics | 已完成 |
| 11 | PPO tanh-squashed Gaussian policy | 已完成 |
| 12 | PPO squashed update epochs tuning | 已完成 |
| 13 | PPO squashed EP4 multi-seed | 已完成 |
| 14 | Stage 2 final baseline video rendering | 已完成/可复用 |
| 15 | Stage 2 final TensorBoard re-run | 暂缓/可复用 |
| Stage 3-01 | SB3 SAC baseline and off-policy comparison | 已完成规划 |
| Stage 3-02 | SB3 SAC smoke test | 当前进行中 |

## 阶段状态

| 阶段 | 主题 | 状态 |
| --- | --- | --- |
| Stage 0 | 项目启动与环境确认 | 已完成 |
| Stage 1 | 普通 PPO baseline | 已完成 |
| Stage 2 | 工程优化与消融 | 已完成 |
| Stage 3 | SAC 强基线与 off-policy 对比 | 进行中 |
| Stage 4 | MaMuJoCo 多智能体扩展 | 后续可选 |
| Stage 5 | 项目总结与简历材料 | 未开始 |

## 当前任务入口

按 `notes/stage3_sac/02_sb3_sac_smoke_test.md` 在 AutoDL 上跑通 SB3 SAC smoke test，并确认 SB3、TensorBoard、checkpoint、VecNormalize 和 evaluation 输出路径。

```text
/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/
```

## 官方资料入口

- [Farama MaMuJoCo Humanoid](https://robotics.farama.org/envs/MaMuJoCo/ma_humanoid/)
- [Farama MaMuJoCo overview](https://robotics.farama.org/envs/MaMuJoCo/)
- [Farama Gymnasium-Robotics installation](https://robotics.farama.org/content/installation/)
