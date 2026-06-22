# AutoDL 主机环境基线

## 用途

本文记录当前 AutoDL 运行环境中相对稳定的事实，用于后续 PPO 训练、实验复现和简历项目说明。

完整原始报告见 `server/autodl_host_report.txt`。本文不记录 SSH/VNC 密码、token、私钥、瞬时进程、瞬时 GPU 利用率等动态或敏感信息。

## 基线快照

记录日期：2026-06-23

| 项目 | 当前配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS |
| CPU | Intel Xeon Platinum 8470, 2 sockets, 52 cores/socket, 208 logical CPUs |
| 内存 | 754 GiB total |
| GPU | NVIDIA GeForce RTX 5090 D, 32607 MiB |
| NVIDIA Driver | 595.71.05 |
| CUDA 兼容版本 | 13.2, from `nvidia-smi` |
| Conda | 24.4.0 |
| 项目 Conda 环境 | `/root/autodl-tmp/conda-envs/humanoid-rl` |
| Python | 3.11.15 |

## 关键 Python 包

| 包 | 版本 |
| --- | --- |
| torch | 2.12.1+cu130 |
| gymnasium | 1.3.0 |
| gymnasium_robotics | 1.4.2 |
| mujoco | 3.9.0 |
| pettingzoo | 1.26.1 |
| tensorboard | 2.20.0 |

## 磁盘

| 路径 | 类型 | 记录时状态 | 用途 |
| --- | --- | --- | --- |
| `/` | 系统盘 overlay | 30G total, 22G used, 8.6G available | 系统和基础软件，避免放训练产物 |
| `/autodl-pub` | 共享/数据挂载 | 7.3T total, 1.6T used, 5.3T available | 公共数据挂载 |
| `/dev/nvme0n1p2` mounted at `/usr/bin/nvidia-smi` | 本地磁盘挂载 | 880G total, 16G used, 820G available | AutoDL 容器挂载 |
| `/dev/shm` | shared memory | 45G total | 可能影响并行采样或大 batch 数据交换 |

项目约定：

- 仓库目录：`/root/autodl-tmp/Humanoid`
- 运行产物目录：建议使用 `/root/autodl-tmp/Humanoid-runs/`
- 不把 checkpoints、TensorBoard events、视频、raw logs 提交到 Git。

## 已确认事项

- MaMuJoCo Humanoid `partitioning=None` smoke test 通过。
- MaMuJoCo Humanoid `partitioning="9|8"` smoke test 通过。
- 当前训练阶段使用无渲染模式；`glxinfo` 没有可用 display 不影响 PPO baseline。
- 后续如需录制视频或可视化，再单独检查 EGL / viewer 配置。

## 不记录的动态信息

- hostname
- SSH/VNC 连接信息
- 密码、token、私钥
- 瞬时 GPU 利用率
- 瞬时进程列表
- 每次训练后的剩余磁盘空间
