# 更新记录

## v0.3.16 - 2026-06-24

- 做了什么：进入 Stage 3 SB3 SAC smoke test；新增 Gymnasium 风格 MaMuJoCo 单智能体 wrapper，新增 `src/train_sac_sb3.py`，新增 `notes/stage3_sac/02_sb3_sac_smoke_test.md`，并更新 README 与路线图的当前入口。
- 为什么：Stage 2 手写 PPO 已经收束，TensorBoard 补充任务暂缓不影响 SAC 推进；先跑通 SB3 SAC 工程链路，再进行 `1M-3M` timesteps 长训和 PPO/SAC 对比。

## v0.3.15 - 2026-06-24

- 做了什么：重组 `notes/` 为 stage 子目录；将 PPO 阶段放入 `notes/stage1_2_ppo/`，将 SAC 规划放入 `notes/stage3_sac/`；新增 PPO final TensorBoard 重跑教程，并为 `src/train_ppo.py` 增加 TensorBoard event 写入。
- 为什么：路线图保留在 `notes/` 根目录做总指挥，阶段教程分目录管理更清晰；SAC 阶段将使用 SB3，不再手写，以便快速建立强 off-policy 对照。

## v0.3.14 - 2026-06-24

- 做了什么：调整后续路线，将 Stage 3 从 MaMuJoCo 多智能体入口改为 SAC baseline 与 off-policy 对比；将多智能体移动到后续可选 Stage 4，并重写 `notes/stage3_sac/01_stage3_sac_entry.md`。
- 为什么：PPO baseline 已经形成完整优化与消融叙事，但视频表现说明单智能体控制质量仍有限；优先引入 SAC 更适合形成强连续控制对照，再考虑多智能体扩展。

## v0.3.13 - 2026-06-24

- 做了什么：新增 `src/render_policy.py`，支持从 PPO checkpoint 录制 deterministic evaluation 视频；新增 `notes/stage1_2_ppo/14_stage2_final_video_rendering.md` 作为当前视频补充任务，并补充视频依赖。
- 为什么：Stage 2 已经得到稳定单智能体 baseline，补充展示视频可以直观看策略行为，帮助判断分数提升是否对应合理控制。

## v0.3.12 - 2026-06-24

- 做了什么：分析 `ppo_long_obsnorm_squash_ep4_seed1/seed2` 真实结果，将 `notes/stage1_2_ppo/13_ppo_squashed_ep4_multiseed.md` 固化为已完成总结，并新增旧版 Stage 3 多智能体入口规划；该入口后续已替换为 SAC 路线。
- 为什么：`observation normalization + tanh-squashed Gaussian policy + update_epochs=4` 完成 seed `0/1/2` 验证，三 seed evaluation mean 为 `825.745`，Stage 2 可以收束并进入 MaMuJoCo 多智能体对比。

## v0.3.11 - 2026-06-24

- 做了什么：重写 `notes/00_project_roadmap.md` 的 Stage 2 说明，将候选方向、中文解释和当前状态合并到同一结构，并补充额外推进方向。
- 为什么：Stage 2 的路线需要直接对应“做了什么、为什么做、当前结论是什么”，避免候选列表和当前状态分离导致阅读困难。

## v0.3.10 - 2026-06-24

- 做了什么：更新 `notes/00_project_roadmap.md` 和 README 的阶段状态，将当前状态修正为 Stage 2 后半段；补充 Stage 2 候选优化方向的中文解释与完成标准。
- 为什么：当前工作已经超出 Stage 1 普通 PPO baseline，进入工程优化、消融和多 seed 稳定性验证，需要让路线图和 README 与真实项目进度一致。

## v0.3.9 - 2026-06-24

- 做了什么：分析 `ppo_long_obsnorm_squash_ep4_seed0` 真实结果，将 `notes/stage1_2_ppo/12_ppo_squashed_update_epochs_tuning.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/13_ppo_squashed_ep4_multiseed.md`。
- 为什么：`update_epochs=4` 在 tanh-squashed policy 上显著提升 return，并降低 KL 与 PPO clip fraction；下一步需要多 seed 验证候选 baseline 是否可靠。

## v0.3.8 - 2026-06-23

- 做了什么：分析 `ppo_long_obsnorm_squash_seed0` 真实结果，将 `notes/stage1_2_ppo/11_ppo_tanh_squashed_policy.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/12_ppo_squashed_update_epochs_tuning.md`。
- 为什么：tanh-squashed policy 将动作裁剪比例降为 `0`，但 `approx_kl` 和 PPO `clip_fraction` 过高，下一步需要降低 update epochs 来减小 update 强度。

## v0.3.7 - 2026-06-23

- 做了什么：分析 `ppo_long_obsnorm_clipdiag_seed0` 真实结果，将 `notes/stage1_2_ppo/10_ppo_action_clipping_diagnostics.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/11_ppo_tanh_squashed_policy.md`；PPO 增加可选 tanh-squashed Gaussian policy。
- 为什么：诊断显示 tail 中约 `98.26%` 的 raw action 维度被环境裁剪，且平均越界幅度约 `19.55`，说明无界 Gaussian policy 与环境动作边界严重不匹配。

## v0.3.6 - 2026-06-23

- 做了什么：统一 `notes/` 中“已完成代码 / 本节代码变化”的写法，按文件列清具体代码改动；README 新增对应维护规则。
- 为什么：阶段总结需要能直接看出改了哪些源码文件、参数、函数和日志列，避免只写概念名导致后续复盘不清楚。

## v0.3.5 - 2026-06-23

- 做了什么：分析 `ppo_long_obsnorm_logstd10_seed0` 真实结果，将 `notes/stage1_2_ppo/09_ppo_relaxed_action_std_control.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/10_ppo_action_clipping_diagnostics.md`；训练日志新增 action clipping diagnostics。
- 为什么：`log_std_max=1.0` 仍导致策略低回报退化，并出现 KL 与 PPO clip fraction 爆炸；下一步应诊断 raw Gaussian action 被环境裁剪的比例。

## v0.3.4 - 2026-06-23

- 做了什么：分析 `ppo_long_obsnorm_logstd05_seed0` 真实结果，将 `notes/stage1_2_ppo/08_ppo_action_std_control.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/09_ppo_relaxed_action_std_control.md`。
- 为什么：`log_std_max=0.5` 成功降低 entropy 和 KL，但策略退化为固定 18 步倒地，说明该上限过紧；下一步测试更宽松的 `log_std_max=1.0`。

## v0.3.3 - 2026-06-23

- 做了什么：分析 `ppo_long_obsnorm_seed0` 真实长训结果，将 `notes/stage1_2_ppo/07_ppo_long_obsnorm_training.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/08_ppo_action_std_control.md`；同时为 PPO 增加可选 action log std clamp 和日志指标。
- 为什么：长训将评估均值提升到 `326.992`，但 entropy、approx KL 和 clip fraction 明显过高，说明动作探索噪声和策略更新幅度需要进一步控制。

## v0.3.2 - 2026-06-23

- 做了什么：分析 `ppo_baseline_v3_obsnorm_kl006_seed0` 真实结果，将 `notes/stage1_2_ppo/06_ppo_kl_target_tuning.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/07_ppo_long_obsnorm_training.md`。
- 为什么：`target_kl=0.06` 比 `0.03` 更合理，但短训表现仍未超过 v1；下一步应使用当前最强配置 `observation normalization + no target_kl` 做 `3M` timesteps 长训练。

## v0.3.1 - 2026-06-23

- 做了什么：分析 `ppo_baseline_v2_obsnorm_kl_seed0` 真实结果，将 `notes/stage1_2_ppo/05_ppo_update_control.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/06_ppo_kl_target_tuning.md`。
- 为什么：`target_kl=0.03` 成功降低 KL 和 clip fraction，但过度限制学习，下一步需要调宽 KL 阈值做对比。

## v0.3.0 - 2026-06-23

- 做了什么：为 PPO update 增加 `--target-kl` early stopping，并在日志中记录 `update_epochs_used` 和 `early_stopped`。
- 为什么：v1 的 observation normalization 有效，但 approx KL 和 clip fraction 过高，需要控制每轮 PPO update 的策略变化幅度。

## v0.2.9 - 2026-06-23

- 做了什么：分析 `ppo_baseline_v1_obsnorm_seed0` 真实结果，将 `notes/stage1_2_ppo/04_ppo_diagnostics_and_obs_norm.md` 固化为已完成总结，并新增 `notes/stage1_2_ppo/05_ppo_update_control.md`。
- 为什么：observation normalization 提升了评估表现并降低 value loss，但 KL 和 clip fraction 过高，下一步需要控制 PPO update 幅度。

## v0.2.8 - 2026-06-23

- 做了什么：实现 observation normalization，checkpoint 保存/加载归一化统计量，并为训练日志增加 rolling episode return 和 rolling episode length。
- 为什么：baseline v0 显示策略有学习趋势但仍不稳定，下一步需要更标准的连续控制输入处理和更可分析的日志。

## v0.2.7 - 2026-06-23

- 做了什么：分析 `ppo_baseline_v0_seed0` 真实实验结果，补充实验记录结论；将 `notes/stage1_2_ppo/03_ppo_baseline_v0.md` 固化为已完成总结；新增 `notes/stage1_2_ppo/04_ppo_diagnostics_and_obs_norm.md` 作为当前任务。
- 为什么：baseline v0 已有基本学习趋势，但 value loss 和波动仍明显，下一步应先做 observation normalization 和日志诊断。

## v0.2.6 - 2026-06-23

- 做了什么：为 `src/` 下 PPO、环境适配、训练、评估和环境检查代码补充中文注释。
- 为什么：代码需要面向初学复盘，关键算法实现位置应能直接对应 PPO 概念。

## v0.2.5 - 2026-06-23

- 做了什么：参考 `参考readme.md` 的最终形态重写项目 README，强化教程推进、实验记录和 Git 管理规则；将 `autodl-project-manager/`、`参考readme.md`、`README copy.md` 排除出项目 Git 管理。
- 为什么：README 应让新对话 AI 直接理解项目推进方式；skill 和参考文档属于本地管理资料，不应作为当前项目源码提交。

## v0.2.4 - 2026-06-23

- 做了什么：将 README 重写为中文项目规则；将 `notes/stage1_2_ppo/02_minimal_ppo_baseline.md` 固化为已完成总结；新增 `notes/stage1_2_ppo/03_ppo_baseline_v0.md` 作为当前任务。
- 为什么：项目应按教程节奏推进，一节完成后总结旧 note，并新建下一节 note；对话只做管理和疑难交流。

## v0.2.3 - 2026-06-23

- 做了什么：新增 `scripts/summarize_ppo_run.py`，并更新 `notes/stage1_2_ppo/02_minimal_ppo_baseline.md`，要求通过 Git 提交轻量实验记录，而不是把训练结果贴回聊天。
- 为什么：项目推进和实验结论应由 notes 与 experiment records 管理；对话只用于管理协调和疑难交流。

## v0.2.2 - 2026-06-23

- 做了什么：更新 `notes/stage1_2_ppo/02_minimal_ppo_baseline.md`，把当前推进任务明确为 `ppo_baseline_v0_seed0` 中等长度训练，并写入运行、评估和回传输出命令。
- 为什么：项目推进应以 notes 为准，对话只做管理和疑难交流。

## v0.2.1 - 2026-06-23

- 做了什么：新增 `experiment_records/ppo_smoke_test_001.md`，记录 PPO smoke test 的训练命令、评估命令、关键指标摘要和观察；同时清理 `notes/stage1_2_ppo/02_minimal_ppo_baseline.md` 中的大段终端输出。
- 为什么：实验结论应由 Git 管理，但只保留足够分析和复现的轻量摘要，大型产物继续留在 AutoDL 数据盘。

## v0.2.0 - 2026-06-23

- 做了什么：新增手写 PPO baseline 初版，包括单智能体环境适配器、Actor-Critic、Gaussian policy、rollout buffer、GAE、PPO clipped update、训练入口和评估入口；同时补充 README 代码风格约定和短训练命令。
- 为什么：进入 Stage 1 的普通 PPO baseline，实现一个教学友好、可逐步调试和扩展的最小训练闭环。

## v0.1.7 - 2026-06-23

- 做了什么：根据服务器提交的 `server/autodl_host_report.txt` 新增 `AUTODL_HOST_BASELINE.md`，并更新 README 与 `notes/stage1_2_ppo/02_minimal_ppo_baseline.md` 的当前状态。
- 为什么：稳定硬件、CUDA、Conda 和关键包版本已经确认，可以作为后续 PPO 实验复现基线。

## v0.1.6 - 2026-06-23

- 做了什么：将 `notes/stage0_setup/01_project_start_and_env_check.md` 更新为已完成结果记录，并新增 `notes/stage1_2_ppo/02_minimal_ppo_baseline.md` 作为下一节普通 PPO 最小训练闭环。
- 为什么：单智能体和 `9|8` 分区 smoke test 都已通过，项目可以从环境检查进入 PPO baseline 阶段。

## v0.1.5 - 2026-06-23

- 做了什么：精简 `notes/stage0_setup/01_project_start_and_env_check.md`，把当前执行命令放到最前面，并将 EGL、viewer、旧 warning 等内容降级为说明。
- 为什么：lesson note 应该是当前任务清单，而不是冗长聊天记录；日常推进以 `notes/` 为准，对话只用于反馈和纠错。

## v0.1.4 - 2026-06-23

- 做了什么：更新 `notes/stage0_setup/01_project_start_and_env_check.md`，记录当前服务器包版本、环境报告文件未生成的原因、拉取 smoke test 修复提交的检查步骤，以及 Adroit warning / OpenGL display 报错的处理判断。
- 为什么：让 lesson note 继续贴合当前真实推进状态，减少重复踩坑。

## v0.1.3 - 2026-06-23

- 做了什么：修正 `src/check_mamujoco_env.py` 的 MaMuJoCo 环境创建方式，改为官方支持的位置参数调用。
- 为什么：`mamujoco_v1.parallel_env` 使用 `parallel_env("Humanoid", partitioning)`，不接受 `domain=` / `task=` / `partitioning=` 这组关键字参数。

## v0.1.2 - 2026-06-23

- 做了什么：将独立 conda 环境路径从 `humanoid-ppo` 改为 `humanoid-rl`。
- 为什么：项目整体目标不止普通 PPO，环境名应覆盖后续优化、多智能体对比和 RL 扩展阶段。

## v0.1.1 - 2026-06-23

- 做了什么：重写 `notes/stage0_setup/01_project_start_and_env_check.md`，使其反映当前状态：远端仓库已连接、服务器应使用数据盘、Humanoid 项目应创建独立 conda 环境、旧 Fetch 环境只用于参考。
- 为什么：`notes/` 中的 lesson 应该实时对应当前要做的事，避免后续按过期步骤操作。

## v0.1.0 - 2026-06-23

- 做了什么：创建项目启动骨架，包括 README、CHANGELOG、notes、AutoDL 检查脚本、依赖清单和 MaMuJoCo 环境冒烟测试脚本。
- 为什么：先建立可持续推进方式，再进入普通 PPO baseline，避免后续训练结果和环境问题混在一起。
- 注意：当前还没有任何训练指标、曲线或性能结论；所有结论必须来自后续真实 AutoDL 输出。
