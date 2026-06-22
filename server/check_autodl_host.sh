#!/usr/bin/env bash
set -euo pipefail

echo "== Operating system =="
cat /etc/os-release || true

echo
echo "== CPU =="
lscpu || true

echo
echo "== Memory =="
free -h || true

echo
echo "== GPU and driver =="
nvidia-smi || true

echo
echo "== Disk =="
df -h || true

echo
echo "== Conda =="
if command -v conda >/dev/null 2>&1; then
  conda --version
  conda env list
else
  echo "conda not found. Check the selected AutoDL image."
fi

echo
echo "== Python =="
python --version || true

echo
echo "== Python packages =="
python - <<'PY' || true
packages = [
    "torch",
    "gymnasium",
    "gymnasium_robotics",
    "mujoco",
    "pettingzoo",
    "tensorboard",
]

for name in packages:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: {version}")
    except Exception as exc:
        print(f"{name}: not available ({exc})")
PY

echo
echo "== OpenGL =="
if command -v glxinfo >/dev/null 2>&1; then
  DISPLAY="${DISPLAY:-:1}" glxinfo -B || true
else
  echo "glxinfo not found. This is only required for some rendering checks."
fi

echo
echo "== MuJoCo render hint =="
echo "For headless rendering later, try MUJOCO_GL=egl on AutoDL when EGL is available."

