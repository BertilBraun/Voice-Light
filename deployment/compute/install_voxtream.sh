#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
voxtream_root="${repository_root}/.cache/compute/voxtream"
source_root="${voxtream_root}/source"
python_path="${voxtream_root}/.venv/bin/python"
revision="8ec2d62159dae4716ae7058827244a962d40603c"

mkdir -p "$voxtream_root"
if [[ ! -d "${source_root}/.git" ]]; then
  git clone https://github.com/herimor/voxtream.git "$source_root"
fi

git -C "$source_root" fetch origin "$revision"
git -C "$source_root" checkout --detach "$revision"
git -C "$source_root" lfs install --local
git -C "$source_root" lfs pull

uv venv --clear --python 3.12 "${voxtream_root}/.venv"
(
  cd "$source_root"
  uv pip install --no-config --python "$python_path" --editable .
)

echo "VoXtream2 environment installed at ${voxtream_root}."
