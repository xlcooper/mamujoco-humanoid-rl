# 02 已完成：SB3 SAC Smoke Test

## 本节目标

Stage 3 从 SB3 SAC smoke test 开始，不再手写 SAC。

本节目标是验证工程链路，而不是追求高分：

- 确认 `stable-baselines3` 在 AutoDL 环境可用。
- 确认 MaMuJoCo `partitioning=None` Humanoid 可以被包装成 Gymnasium 单智能体环境。
- 跑通一个很短的 SB3 SAC 训练入口。
- 确认 SB3 `Monitor`、TensorBoard、checkpoint、VecNormalize 和 evaluation 输出路径。
- 生成 Git 管理的轻量实验记录。

## 本节代码变化

1. `src/envs.py`

   - 新增 `GymnasiumSingleAgentMaMuJoCoEnv`。
   - 新增 `make_humanoid_gymnasium_env()`。
   - 将 MaMuJoCo PettingZoo Parallel API 的单智能体环境转换成 Gymnasium `reset()` / `step()` 风格：
     - `reset()` 返回 `(observation, info)`。
     - `step()` 返回 `(observation, reward, terminated, truncated, info)`。
   - 保留原有 `SingleAgentMaMuJoCoEnv` 和 `make_humanoid_single_agent_env()`，不影响手写 PPO 训练和评估。

2. `src/train_sac_sb3.py`

   - 新增 SB3 SAC 训练入口。
   - 使用 `SAC(MlpPolicy)`，默认 `ent_coef="auto"` 和 `target_entropy="auto"`。
   - 使用 SB3 `Monitor` 记录 episode return / length。
   - 默认使用 SB3 `VecNormalize(norm_obs=True, norm_reward=False)` 做 observation normalization。
   - 写入 `config.json`、TensorBoard event、checkpoint、`vecnormalize.pkl`、`eval_results.json` 和 `eval_output.txt`。
   - 支持 `--no-normalize-observations` 做后续消融。

3. `src/evaluate_sac_sb3.py`

   - 新增 SB3 SAC checkpoint 评估入口。
   - 加载 `checkpoints/sac_final.zip`。
   - 如训练启用了 observation normalization，则通过 `--vecnormalize` 加载 `vecnormalize.pkl`。
   - 使用 deterministic action 输出 episode return、episode length、mean return、std return 和 mean length。
   - 可用 `--output-json` 保存结构化评估结果。

4. `scripts/summarize_sac_run.py`

   - 新增 SAC run 轻量摘要脚本。
   - 读取 `config.json`、SB3 `monitor.monitor.csv`、评估文本和评估 JSON。
   - 生成 `experiment_records/sac_sb3_smoke_seed0.md` 这类可提交到 Git 的轻量实验记录。

## 训练配置

实验记录：

- `experiment_records/sac_sb3_smoke_seed0.md`

核心配置：

- seed: `0`
- total timesteps: `5000`
- learning starts: `500`
- replay buffer size: `100000`
- batch size: `256`
- train frequency: `1`
- gradient steps: `1`
- entropy coefficient: `auto`
- target entropy: `auto`
- observation normalization: SB3 `VecNormalize(norm_obs=True, norm_reward=False)`
- network: `MlpPolicy`, `net_arch=[256, 256]`

训练命令：

```bash
python src/train_sac_sb3.py \
  --seed 0 \
  --total-timesteps 5000 \
  --learning-starts 500 \
  --batch-size 256 \
  --buffer-size 100000 \
  --run-name sac_sb3_smoke_seed0 \
  --eval-episodes 3
```

## 真实结果

训练结束标志：

```text
training_done=true run_dir=/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0
```

训练末段 SB3 日志摘要：

```text
rollout/ep_len_mean     40.4
rollout/ep_rew_mean     186
time/total_timesteps    4899
train/actor_loss        -147
train/critic_loss       7.21
train/ent_coef          0.29
train/ent_coef_loss     -23.3
train/n_updates         4398
```

deterministic evaluation：

```text
episode=1 return=198.458 length=41
episode=2 return=217.913 length=45
episode=3 return=204.797 length=42
mean_return=207.056 std_return=8.102
mean_length=42.667
```

## 本节分析

- SB3 SAC 训练入口已跑通，训练过程中出现 rollout 与 train 指标。
- `Monitor` 正常记录 episode return / length；tail 中 episode return 最高达到 `550.734`，episode length 最高达到 `114`。
- `checkpoints/sac_final.zip`、`vecnormalize.pkl`、`eval_results.json`、`eval_output.txt` 和 TensorBoard 目录都按预期生成。
- deterministic evaluation 能加载 checkpoint 和 VecNormalize 统计量并完成 3 episodes 评估。
- 5000 timesteps 太短，只能验证工程链路；`207.056` 的 evaluation mean return 不作为 SAC 性能结论。

## 本节结论

Stage 3 SB3 SAC smoke test 通过。

这说明：

- MaMuJoCo Humanoid 单智能体 Gymnasium wrapper 可用于 SB3。
- SB3 SAC、Monitor、TensorBoard、checkpoint、VecNormalize 和 evaluation 链路可以工作。
- 可以进入更长的 SAC baseline 训练。

## 下一节

进入：

- `notes/stage3_sac/03_sb3_sac_long_training.md`

下一节目标：

1. 跑 `1M` timesteps 的 SB3 SAC seed `0` baseline。
2. 记录 TensorBoard、Monitor、evaluation 和 wall-clock。
3. 与 PPO final baseline 的 `825.745` three-seed mean 建立第一轮对比。
