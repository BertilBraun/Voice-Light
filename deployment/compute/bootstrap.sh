#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap.sh requires Ubuntu Linux." >&2
  exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
  sudo_command=()
elif command -v sudo >/dev/null 2>&1; then
  sudo_command=(sudo)
else
  echo "Run as root or install sudo so system packages can be installed." >&2
  exit 1
fi

"${sudo_command[@]}" apt-get update
"${sudo_command[@]}" apt-get install -y \
  build-essential \
  ca-certificates \
  clang \
  curl \
  ffmpeg \
  git \
  git-lfs \
  iproute2 \
  libsndfile1 \
  libsndfile1-dev \
  libsox-dev \
  pkg-config \
  sox

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing. Rent an NVIDIA CUDA instance before bootstrapping." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation completed but uv is not on PATH." >&2
  exit 1
fi

uv python install 3.12
uv sync --frozen --python 3.12

mkdir -p .cache/compute/huggingface .cache/compute/torch logs/compute run/compute
if [[ ! -f .env.compute ]]; then
  .venv/bin/python deployment/compute/configure_environment.py "$repository_root"
  echo "Created .env.compute with a new bearer token. Copy that token to the local app securely."
fi

set -a
source .env.compute
set +a
.venv/bin/python deployment/compute/validate_environment.py --download-models

echo "Bootstrap complete. Start the backend with: bash deployment/compute/start.sh"
