# 02 当前任务：SB3 SAC Smoke Test

## 本节目标

Stage 3 从 SB3 SAC smoke test 开始，不再手写 SAC。

本节目标是先验证工程链路，而不是追求高分：

- 确认 `stable-baselines3` 在 AutoDL 环境可用。
- 确认 MaMuJoCo `partitioning=None` Humanoid 可以被包装成 Gymnasium 单智能体环境。
- 跑通一个很短的 SB3 SAC 训练入口。
- 确认 SB3 `Monitor`、TensorBoard、checkpoint、VecNormalize 和 evaluation 输出路径。

注意：smoke test 的 return 只用于确认链路是否正常，不作为 SAC 性能结论。

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

## AutoDL 依赖确认

在服务器运行：

```bash
cd /root/autodl-tmp/Humanoid
git pull --rebase
conda activate /root/autodl-tmp/conda-envs/humanoid-rl
pip install -r requirements.txt
```

确认 SB3 和 Gymnasium 依赖：

```bash
python - <<'PY'
import stable_baselines3 as sb3
import gymnasium
import gymnasium_robotics
import mujoco

print("stable_baselines3", sb3.__version__)
print("gymnasium", gymnasium.__version__)
print("gymnasium_robotics ok")
print("mujoco", mujoco.__version__)
PY
```

## Smoke Test 训练命令

先跑一个短训版本：

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

这条训练命令会在结束时自动做一次短评估，并写入：

```text
/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/eval_results.json
/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/eval_output.txt
```

为了保持和 Stage 1-2 一样的实验管理方式，训练结束后仍建议单独复评一次，并把输出保存下来。

如果 `5000` timesteps 太慢，可以先降到 `1000`，但 `learning-starts` 也要同步降到 `100`：

```bash
python src/train_sac_sb3.py \
  --seed 0 \
  --total-timesteps 1000 \
  --learning-starts 100 \
  --batch-size 128 \
  --buffer-size 50000 \
  --run-name sac_sb3_smoke_seed0_fast \
  --eval-episodes 2
```

## 训练结束后评估

默认 smoke test 使用 observation normalization，因此评估时需要加载 `vecnormalize.pkl`：

```bash
python src/evaluate_sac_sb3.py \
  --checkpoint /root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/checkpoints/sac_final.zip \
  --vecnormalize /root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/vecnormalize.pkl \
  --episodes 3 \
  --output-json /root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/eval_results.json \
  | tee /root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/eval_output.txt
```

如果运行 smoke test 时使用了 `--no-normalize-observations`，则评估命令去掉 `--vecnormalize ...`。

## 生成 Git 管理的轻量记录

训练和评估完成后，在服务器生成实验记录：

```bash
python scripts/summarize_sac_run.py \
  --run-dir /root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0 \
  --eval-output /root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/eval_output.txt \
  --eval-json /root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/eval_results.json \
  --output experiment_records/sac_sb3_smoke_seed0.md
```

只提交轻量记录，不提交 run 目录、checkpoint、TensorBoard event 或 replay buffer：

```bash
git add experiment_records/sac_sb3_smoke_seed0.md
git commit -m "Record SB3 SAC smoke test seed0 summary"
git pull --rebase
git push
```

如果训练或评估报错，把完整 traceback 贴回对话；如果没报错，就通过 Git 管理结果。我本地 pull 后再分析 smoke test 是否通过，并把本 note 固化为已完成总结。

## 输出路径

默认 run 目录：

```text
/root/autodl-tmp/Humanoid-runs/sac_sb3_smoke_seed0/
```

需要确认的关键产物：

```text
config.json
monitor.monitor.csv
tensorboard/
checkpoints/sac_final.zip
vecnormalize.pkl
eval_results.json
eval_output.txt
```

TensorBoard 启动方式：

```bash
tensorboard \
  --logdir /root/autodl-tmp/Humanoid-runs \
  --host 0.0.0.0 \
  --port 6006
```

优先看：

- `rollout/ep_rew_mean`
- `rollout/ep_len_mean`
- `train/actor_loss`
- `train/critic_loss`
- `train/ent_coef`
- `train/ent_coef_loss`

## 本地验证状态

Windows 本机当前没有安装项目运行依赖，`numpy`、`gymnasium`、`stable-baselines3`、`gymnasium_robotics` 和 `mujoco` 都不可用。

已完成的本地轻量验证：

```bash
python -m py_compile src/envs.py src/train_sac_sb3.py src/evaluate_sac_sb3.py scripts/summarize_sac_run.py
```

真实 smoke test 仍必须在 AutoDL 环境运行，不能把本地编译通过当作训练结论。

## Smoke Test 通过标准

- `stable-baselines3`、`gymnasium_robotics` 和 `mujoco` 可以正常 import。
- `src/train_sac_sb3.py` 可以创建 Humanoid 环境并开始训练。
- 训练过程中出现 SB3 rollout / train 日志。
- run 目录下生成 `config.json`、`monitor.monitor.csv`、TensorBoard event、`checkpoints/sac_final.zip`。
- 默认 observation normalization 开启时生成 `vecnormalize.pkl`。
- 训练结束后生成 `eval_results.json` 和 `eval_output.txt`。
- `scripts/summarize_sac_run.py` 能生成 `experiment_records/sac_sb3_smoke_seed0.md`。
- 服务器完成轻量记录 commit、`git pull --rebase` 和 push。

## 跑完后如何固化

如果 AutoDL smoke test 通过，下一步：

1. 本地 `git pull --rebase` 读取 `experiment_records/sac_sb3_smoke_seed0.md`。
2. 分析 `monitor.monitor.csv` tail、evaluation return/length、TensorBoard 关键曲线是否正常。
3. 将本 note 改成“已完成总结”。
4. 新建下一节 `03_sb3_sac_long_training.md` 或先调整 smoke test 参数，视真实结果决定。

如果 smoke test 报错，把完整报错贴回对话；默认按 AutoDL 环境问题处理。
