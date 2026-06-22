# AutoDL Reference

Use this reference when setting up AutoDL-backed projects or summarizing environment reports.

## Default Workflow

Assume this split unless the user says otherwise:

- Local machine: editing, reading, Git, project summaries.
- AutoDL server: dependency install, training, evaluation, rendering, TensorBoard, long-running jobs.

Do not infer server capabilities from the local machine.

## Common AutoDL Conventions

Typical paths:

```text
/root/autodl-tmp/                 data disk; use for repos and run artifacts
/root/autodl-tmp/<project-data>/  models/logs/tensorboard/eval/videos
/root/miniconda3/envs/            conda envs
```

Common rules:

- Keep the system disk light.
- Put generated models, logs, TensorBoard events, videos, and datasets on the data disk.
- Do not commit SSH host/port, VNC password, private keys, tokens, or account details.
- Use EGL for offscreen MuJoCo rendering when appropriate.
- Save stable environment facts, not transient process status.

## Environment Check Flow

1. Copy `scripts/check_autodl_host.sh` from this skill to `server/check_autodl_host.sh`.
2. Ask the user to run:

```bash
bash server/check_autodl_host.sh | tee server/autodl_host_report.txt
git add server/check_autodl_host.sh server/autodl_host_report.txt
git commit -m "Record AutoDL host report"
git pull --rebase
git push
```

3. Pull locally.
4. Read `server/autodl_host_report.txt`.
5. Write `AUTODL_HOST_BASELINE.md`.
6. Remove or ignore raw reports later if they become noisy, but keep the stable baseline.

## AUTODL_HOST_BASELINE.md Template

Recommended structure:

```text
# AutoDL 主机环境基线

## 用途

说明本文件记录稳定环境配置。完整命令输出只在需要时重新生成。

## 基线快照

记录日期：

| 项目 | 当前配置 |
| --- | --- |
| 操作系统 | ... |
| CPU | ... |
| 内存 | ... |
| GPU | ... |
| NVIDIA 驱动 | ... |
| CUDA 兼容版本 | ... |
| Conda | ... |
| 项目 Conda 环境 | ... |

## 磁盘

| 路径 | 类型 | 记录时状态 | 用途 |
| --- | --- | --- | --- |
| / | 系统盘 | ... | 软件和环境 |
| /root/autodl-tmp | 数据盘 | ... | 仓库和运行产物 |

## 已确认事项

- ...

## 不记录的动态信息

- hostname
- PID
- 瞬时 GPU 利用率
- SSH/VNC 密码和端口
- 私钥/token
```

## When to Update the Baseline

Update only when stable environment facts change:

- New instance type.
- GPU/CPU/memory changed.
- Conda environment changed materially.
- Disk layout changed.
- Rendering/VNC/EGL configuration changed.

Do not update just because transient GPU utilization, process list, or free space changed after a run.
