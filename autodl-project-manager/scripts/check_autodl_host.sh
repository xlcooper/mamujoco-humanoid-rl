#!/usr/bin/env bash
set -euo pipefail

echo "== 操作系统 =="
cat /etc/os-release || true

echo
echo "== CPU =="
lscpu || true

echo
echo "== 内存 =="
free -h || true

echo
echo "== GPU 和驱动 =="
nvidia-smi || true

echo
echo "== 磁盘 =="
df -h || true

echo
echo "== Conda =="
if command -v conda >/dev/null 2>&1; then
  conda --version
  conda env list
else
  echo "未找到 conda。请先确认 AutoDL 基础镜像是否包含 Conda。"
fi

echo
echo "== Python =="
python --version || true

echo
echo "== Python packages =="
python - <<'PY' || true
packages = ["torch", "gymnasium", "gymnasium_robotics", "mujoco", "stable_baselines3"]
for name in packages:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: {version}")
    except Exception as exc:
        print(f"{name}: not available ({exc})")
PY

echo
echo "== VNC =="
if command -v tigervncserver >/dev/null 2>&1; then
  tigervncserver -list || true
else
  echo "未找到 tigervncserver。运行 bash server/setup_autodl_mujoco.sh 安装。"
fi

echo
echo "== OpenGL =="
if command -v glxinfo >/dev/null 2>&1; then
  DISPLAY="${DISPLAY:-:1}" glxinfo -B || true
else
  echo "未找到 glxinfo。运行 bash server/setup_autodl_mujoco.sh 安装。"
fi
