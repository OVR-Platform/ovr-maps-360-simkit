#!/usr/bin/env bash
# Isaac Sim, headless, for the cross-engine drop test (simkit.physics.isaac_drop).
# Kept apart from the simkit env: Isaac pins its own numpy and USD.
#
#   bash install/install_env_isaac.sh             # env name: simkit-isaac
#   SIMKIT_ISAAC_ENV=myenv ISAACSIM_VERSION=5.1.0 bash install/install_env_isaac.sh
#
# Requirements: NVIDIA driver >= 535 with a CUDA GPU, glibc >= 2.35
# (Ubuntu 22.04+), ~25 GB of disk for the wheels and the extension cache.
# Installing means accepting the NVIDIA Isaac Sim EULA; this script accepts it
# non-interactively (OMNI_KIT_ACCEPT_EULA=YES), so read it first:
# https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html
set -euo pipefail
ENV_NAME="${SIMKIT_ISAAC_ENV:-simkit-isaac}"
VERSION="${ISAACSIM_VERSION:-5.1.0}"

command -v conda >/dev/null || { echo "conda not found on PATH" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: Isaac Sim needs an NVIDIA GPU" >&2; exit 1; }
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==${VERSION}" --extra-index-url https://pypi.nvidia.com

# First start builds the extension cache (several minutes); do it here, not
# in the middle of a batch.
OMNI_KIT_ACCEPT_EULA=YES python - <<'PY'
from isaacsim.simulation_app import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.core.api import SimulationContext  # noqa: F401
print("simkit-isaac env ok")
app.close()
PY
echo "Isaac interpreter: $(command -v python)  (simkit finds it by env name, or set SIMKIT_ISAAC_PYTHON)"
