#!/usr/bin/env bash
# setup_runpod.sh — Reproducible install of prime-rl on RunPod
# Usage: bash setup_runpod.sh
set -euo pipefail

REPO_URL="https://github.com/PrimeIntellect-ai/prime-rl.git"
INSTALL_DIR="/root/prime-rl"

echo "=== [1/5] System dependencies ==="
apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ninja-build software-properties-common

echo "=== [2/5] Install Python 3.12 if not present ==="
if ! python3.12 --version &>/dev/null; then
    # Add deadsnakes PPA manually (add-apt-repository is broken on some RunPod images)
    echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
        > /etc/apt/sources.list.d/deadsnakes.list
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776
    apt-get update
    apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv
fi
python3.12 --version

echo "=== [3/5] Install uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR="/usr/local/bin" sh
fi
uv --version

echo "=== [4/5] Clone prime-rl ==="
if [ -d "$INSTALL_DIR" ]; then
    echo "Directory $INSTALL_DIR already exists, pulling latest..."
    cd "$INSTALL_DIR" && git pull
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo "=== [5/5] Install prime-rl with uv ==="
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
uv sync --all-extras --locked --no-dev

echo ""
echo "=== Installation complete ==="
echo "To use: cd $INSTALL_DIR && uv run <command>"
echo "Example: uv run rl @ configs/example.toml"
