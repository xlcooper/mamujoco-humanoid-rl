# 更新记录

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
