#!/usr/bin/env bash
# Environment for the whole build: frame, navmesh, CoACD, MJCF/USD export,
# MuJoCo gate, LOD, certification. CPU only.
#
#   bash install/install_env_simkit.sh            # env name: simkit
#   SIMKIT_ENV=myenv bash install/install_env_simkit.sh
set -euo pipefail
ENV_NAME="${SIMKIT_ENV:-simkit}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v conda >/dev/null || { echo "conda not found on PATH" >&2; exit 1; }
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env $ENV_NAME exists, updating it"
else
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -r "$REPO/requirements.txt"
python -m pip install -e "$REPO"

# Import check, in the order that matters: pxr before coacd (see README).
python - <<'PY'
from pxr import Usd
import coacd, cv2, mujoco, open3d, trimesh, scipy
print("simkit env ok: mujoco", mujoco.__version__, "| open3d", open3d.__version__, "| usd", Usd.GetVersion())
PY
