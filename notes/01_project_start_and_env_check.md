# 01 项目启动与环境检查

## 本节目标

这一节只做三件事：

1. 确认 AutoDL 环境信息。
2. 安装最小依赖。
3. 确认 MaMuJoCo Humanoid 可以 reset 和 step。

先不开始 PPO 训练。强化学习项目里，环境、依赖和渲染问题很容易伪装成算法问题，所以第一节先把地基踩实。

## 你需要在 AutoDL 上执行

建议项目放在数据盘，例如：

```bash
cd /root/autodl-tmp/Humanoid
```

采集环境报告：

```bash
bash server/check_autodl_host.sh | tee server/autodl_host_report.txt
```

如果当前项目已经连接 Git 远端，再提交报告：

```bash
git add server/check_autodl_host.sh server/autodl_host_report.txt
git commit -m "Record AutoDL host report"
git pull --rebase
git push
```

如果还没有 Git 远端，就先把 `server/autodl_host_report.txt` 的关键输出贴回来。

## 创建 Python 环境

建议先用 Python 3.11：

```bash
conda create -n humanoid-ppo python=3.11 -y
conda activate humanoid-ppo
pip install -r requirements.txt
```

Farama 官方安装文档推荐安装 `gymnasium-robotics`，MaMuJoCo 依赖 MuJoCo 和 PettingZoo 风格接口。本项目的 `requirements.txt` 先保持宽松，等 AutoDL 报告真实版本后再决定是否锁版本。

参考：<https://robotics.farama.org/content/installation/>

## 运行冒烟测试

先测单智能体版本：

```bash
python src/check_mamujoco_env.py --partitioning none --steps 5
```

再测两智能体分区版本：

```bash
python src/check_mamujoco_env.py --partitioning "9|8" --steps 5
```

预期现象：

- 脚本能打印 agents。
- 能打印 observation/action space。
- 随机动作能连续 step 几步。
- 不要求 reward 好看，因为现在只是随机策略。

## 回传给我

完成后，把下面信息贴回来：

- `server/check_autodl_host.sh` 的 GPU、CUDA、Python、关键包版本。
- 单智能体 smoke test 是否通过。
- `9|8` 分区 smoke test 是否通过。
- 如果报错，贴完整 traceback 的最后 80 行左右。

## 本节完成标准

- AutoDL 环境信息已确认。
- `python src/check_mamujoco_env.py --partitioning none --steps 5` 成功。
- 对 `partitioning="9|8"` 的可用性有明确结果。
- 下一节可以开始普通 PPO 的代码结构设计。

