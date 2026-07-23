#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_path="$repository_root/.venv/bin/python"

torch_version="2.7.1"
torchvision_version="0.22.1"
cuda_wheel_index="https://download.pytorch.org/whl/cu126"

if [[ ! -x "$python_path" ]]; then
  echo "The compute virtual environment is missing. Run uv sync before installing the ASR runtime." >&2
  exit 1
fi

uv pip install --no-config --python "$python_path" --reinstall \
  "torch==${torch_version}" \
  "torchvision==${torchvision_version}" \
  "torchaudio==${torch_version}" \
  --index-url "$cuda_wheel_index"

"$python_path" -c '
import torch
import torchaudio
import torchvision

if torch.version.cuda != "12.6":
    raise RuntimeError(f"Expected a CUDA 12.6 PyTorch runtime, got {torch.version.cuda!r}.")
print(
    "Installed ASR CUDA runtime: "
    f"torch={torch.__version__}, torchaudio={torchaudio.__version__}, "
    f"torchvision={torchvision.__version__}, CUDA={torch.version.cuda}."
)
'
