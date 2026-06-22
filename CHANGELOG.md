# 更新记录

## v0.2.6 - 2026-06-23

- 做了什么：为 `src/` 下 PPO、环境适配、训练、评估和环境检查代码补充中文注释。
- 为什么：代码需要面向初学复盘，关键算法实现位置应能直接对应 PPO 概念。

## v0.2.5 - 2026-06-23

- 做了什么：参考 `参考readme.md` 的最终形态重写项目 README，强化教程推进、实验记录和 Git 管理规则；将 `autodl-project-manager/`、`参考readme.md`、`README copy.md` 排除出项目 Git 管理。
- 为什么：README 应让新对话 AI 直接理解项目推进方式；skill 和参考文档属于本地管理资料，不应作为当前项目源码提交。

## v0.2.4 - 2026-06-23

- 做了什么：将 README 重写为中文项目规则；将 `notes/02_minimal_ppo_baseline.md` 固化为已完成总结；新增 `notes/03_ppo_baseline_v0.md` 作为当前任务。
- 为什么：项目应按教程节奏推进，一节完成后总结旧 note，并新建下一节 note；对话只做管理和疑难交流。

## v0.2.3 - 2026-06-23

- 做了什么：新增 `scripts/summarize_ppo_run.py`，并更新 `notes/02_minimal_ppo_baseline.md`，要求通过 Git 提交轻量实验记录，而不是把训练结果贴回聊天。
- 为什么：项目推进和实验结论应由 notes 与 experiment records 管理；对话只用于管理协调和疑难交流。

## v0.2.2 - 2026-06-23

- 做了什么：更新 `notes/02_minimal_ppo_baseline.md`，把当前推进任务明确为 `ppo_baseline_v0_seed0` 中等长度训练，并写入运行、评估和回传输出命令。
- 为什么：项目推进应以 notes 为准，对话只做管理和疑难交流。

## v0.2.1 - 2026-06-23

- 做了什么：新增 `experiment_records/ppo_smoke_test_001.md`，记录 PPO smoke test 的训练命令、评估命令、关键指标摘要和观察；同时清理 `notes/02_minimal_ppo_baseline.md` 中的大段终端输出。
- 为什么：实验结论应由 Git 管理，但只保留足够分析和复现的轻量摘要，大型产物继续留在 AutoDL 数据盘。

## v0.2.0 - 2026-06-23

- 做了什么：新增手写 PPO baseline 初版，包括单智能体环境适配器、Actor-Critic、Gaussian policy、rollout buffer、GAE、PPO clipped update、训练入口和评估入口；同时补充 README 代码风格约定和短训练命令。
- 为什么：进入 Stage 1 的普通 PPO baseline，实现一个教学友好、可逐步调试和扩展的最小训练闭环。

## v0.1.7 - 2026-06-23

- 做了什么：根据服务器提交的 `server/autodl_host_report.txt` 新增 `AUTODL_HOST_BASELINE.md`，并更新 README 与 `notes/02_minimal_ppo_baseline.md` 的当前状态。
- 为什么：稳定硬件、CUDA、Conda 和关键包版本已经确认，可以作为后续 PPO 实验复现基线。

## v0.1.6 - 2026-06-23

- 做了什么：将 `notes/01_project_start_and_env_check.md` 更新为已完成结果记录，并新增 `notes/02_minimal_ppo_baseline.md` 作为下一节普通 PPO 最小训练闭环。
- 为什么：单智能体和 `9|8` 分区 smoke test 都已通过，项目可以从环境检查进入 PPO baseline 阶段。

## v0.1.5 - 2026-06-23

- 做了什么：精简 `notes/01_project_start_and_env_check.md`，把当前执行命令放到最前面，并将 EGL、viewer、旧 warning 等内容降级为说明。
- 为什么：lesson note 应该是当前任务清单，而不是冗长聊天记录；日常推进以 `notes/` 为准，对话只用于反馈和纠错。

## v0.1.4 - 2026-06-23

- 做了什么：更新 `notes/01_project_start_and_env_check.md`，记录当前服务器包版本、环境报告文件未生成的原因、拉取 smoke test 修复提交的检查步骤，以及 Adroit warning / OpenGL display 报错的处理判断。
- 为什么：让 lesson note 继续贴合当前真实推进状态，减少重复踩坑。

## v0.1.3 - 2026-06-23

- 做了什么：修正 `src/check_mamujoco_env.py` 的 MaMuJoCo 环境创建方式，改为官方支持的位置参数调用。
- 为什么：`mamujoco_v1.parallel_env` 使用 `parallel_env("Humanoid", partitioning)`，不接受 `domain=` / `task=` / `partitioning=` 这组关键字参数。

## v0.1.2 - 2026-06-23

- 做了什么：将独立 conda 环境路径从 `humanoid-ppo` 改为 `humanoid-rl`。
- 为什么：项目整体目标不止普通 PPO，环境名应覆盖后续优化、多智能体对比和 RL 扩展阶段。

## v0.1.1 - 2026-06-23

- 做了什么：重写 `notes/01_project_start_and_env_check.md`，使其反映当前状态：远端仓库已连接、服务器应使用数据盘、Humanoid 项目应创建独立 conda 环境、旧 Fetch 环境只用于参考。
- 为什么：`notes/` 中的 lesson 应该实时对应当前要做的事，避免后续按过期步骤操作。

## v0.1.0 - 2026-06-23

- 做了什么：创建项目启动骨架，包括 README、CHANGELOG、notes、AutoDL 检查脚本、依赖清单和 MaMuJoCo 环境冒烟测试脚本。
- 为什么：先建立可持续推进方式，再进入普通 PPO baseline，避免后续训练结果和环境问题混在一起。
- 注意：当前还没有任何训练指标、曲线或性能结论；所有结论必须来自后续真实 AutoDL 输出。
