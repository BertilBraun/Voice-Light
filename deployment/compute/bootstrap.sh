#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

deployment_mode="full"
if [[ "$#" -eq 2 && "$1" == "--mode" ]]; then
  deployment_mode="$2"
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: bootstrap.sh [--mode full|asr]" >&2
  exit 1
fi
if [[ "$deployment_mode" != "full" && "$deployment_mode" != "asr" ]]; then
  echo "Deployment mode must be either 'full' or 'asr'." >&2
  exit 1
fi

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
system_packages=(
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  git-lfs \
  libsndfile1 \
  libsndfile1-dev \
  pkg-config \
  sox
)
if [[ "$deployment_mode" == "full" ]]; then
  system_packages+=(clang espeak-ng iproute2 libsox-dev)
fi
"${sudo_command[@]}" apt-get install -y "${system_packages[@]}"

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
if [[ "$deployment_mode" == "asr" ]]; then
  uv sync --frozen --python 3.12 --extra compute \
    --no-install-package moshi --no-install-package peft \
    --no-install-package torch --no-install-package torchaudio \
    --no-install-package torchvision
  bash deployment/compute/install_asr_torch_cu126.sh
else
  uv sync --frozen --python 3.12 --extra compute
  bash deployment/compute/install_vllm.sh
fi

mkdir -p .cache/compute/huggingface .cache/compute/torch logs/compute run/compute
if [[ ! -f .env.compute ]]; then
  .venv/bin/python deployment/compute/configure_environment.py \
    "$repository_root" --mode "$deployment_mode"
  echo "Created .env.compute with a new bearer token. Copy that token to the local app securely."
else
  voice_stack_enabled="true"
  if [[ "$deployment_mode" == "asr" ]]; then
    voice_stack_enabled="false"
  fi
  if grep -q '^VOICE_LIGHT_VOICE_STACK_ENABLED=' .env.compute; then
    sed -i "s/^VOICE_LIGHT_VOICE_STACK_ENABLED=.*/VOICE_LIGHT_VOICE_STACK_ENABLED=$voice_stack_enabled/" \
      .env.compute
  else
    printf '\nVOICE_LIGHT_VOICE_STACK_ENABLED=%s\n' "$voice_stack_enabled" >> .env.compute
  fi
fi

set -a
source .env.compute
set +a
if [[ "$deployment_mode" == "full" && "${VOICE_LIGHT_TTS_BACKEND:-kyutai}" == "voxtream" ]]; then
  bash deployment/compute/install_voxtream.sh
fi
PYTHONPATH="$repository_root" \
  .venv/bin/python deployment/compute/validate_environment.py \
    --mode "$deployment_mode" --download-models

echo "Bootstrap complete for $deployment_mode mode. Start the backend with: bash deployment/compute/start.sh"
