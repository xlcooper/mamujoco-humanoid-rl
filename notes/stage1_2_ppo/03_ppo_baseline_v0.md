# 03 已完成：PPO Baseline v0

## 本节目标

在 `partitioning=None` 的 MaMuJoCo Humanoid 上运行第一条可分析的 PPO baseline，判断当前手写 PPO 是否有基本学习趋势，以及是否暴露稳定性问题。

## 已知基础

- 环境检查已完成：`notes/stage0_setup/01_project_start_and_env_check.md`
- 最小训练闭环已完成：`notes/stage1_2_ppo/02_minimal_ppo_baseline.md`
- AutoDL 基线已记录：`AUTODL_HOST_BASELINE.md`
- Smoke test 记录：`experiment_records/ppo_smoke_test_001.md`

## 训练配置

实验记录：

- `experiment_records/ppo_baseline_v0_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `100000`
- rollout steps: `2048`
- batch size: `256`
- update epochs: `10`
- learning rate: `3e-4`
- hidden size: `256`
- observation dim: `348`
- action dim: `17`

实际训练步数是 `98304`，因为当前代码按 `total_timesteps // rollout_steps` 计算 update 数。

## 评估结果

5 episode evaluation：

```text
episode=1 return=249.248 length=47
episode=2 return=228.679 length=44
episode=3 return=246.010 length=47
episode=4 return=248.831 length=47
episode=5 return=247.117 length=47
mean_return=243.977 std_return=7.738
```

## 本节分析

- baseline v0 明显优于 smoke test，但还不是稳定策略。
- 训练末段 episode return 波动较大，tail 中最低约 `130.809`，最高约 `477.083`。
- episode length 多次达到 60-100 step，说明策略开始学到延长存活时间。
- value loss 没有爆炸，末段从 500 左右下降到约 `267.628`，但 critic 压力仍明显。
- entropy 没有塌缩，探索暂时不是主要问题。
- approx KL 多数在 `0.02` 附近，clip fraction 常在 `0.19` 到 `0.26`，提示 PPO update 幅度偏大，需要后续观察。

## 本节结论

- 手写 PPO baseline 有基本学习趋势。
- 100k steps 不足以作为最终表现结论。
- 下一步应先做诊断和稳定性增强，而不是直接堆训练步数。

## 下一节

进入：

- `notes/stage1_2_ppo/04_ppo_diagnostics_and_obs_norm.md`

下一节目标：

1. 增加 observation normalization。
2. 改善训练日志，让本地分析更方便。
3. 跑 PPO baseline v1，与 v0 对比。
