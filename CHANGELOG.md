# 更新记录

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
