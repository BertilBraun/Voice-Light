#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
voxtream_root="${repository_root}/.cache/compute/voxtream"
source_root="${voxtream_root}/source"
python_path="${voxtream_root}/.venv/bin/python"
revision="8ec2d62159dae4716ae7058827244a962d40603c"
prompt_cache_patch="${repository_root}/deployment/compute/patches/voxtream-prompt-memory-cache.patch"

mkdir -p "$voxtream_root"
if [[ ! -d "${source_root}/.git" ]]; then
  git clone https://github.com/herimor/voxtream.git "$source_root"
fi

git -C "$source_root" fetch origin "$revision"
git -C "$source_root" checkout --detach "$revision"
if ! git -C "$source_root" apply --reverse --check "$prompt_cache_patch" >/dev/null 2>&1; then
  git -C "$source_root" apply --check "$prompt_cache_patch"
  git -C "$source_root" apply "$prompt_cache_patch"
fi
git -C "$source_root" lfs install --local
git -C "$source_root" lfs pull

if [[ ! -x "$python_path" ]] || ! "$python_path" -c "import torch; import torchaudio"; then
  uv venv --clear --python 3.12 "${voxtream_root}/.venv"
fi
(
  cd "$source_root"
  uv pip install --no-config --python "$python_path" --editable .
)

echo "VoXtream2 environment installed at ${voxtream_root}."
