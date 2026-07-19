#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: provision-vast.sh <bundle-path> <repository-path> <branch>" >&2
  exit 1
fi

bundle_path="$1"
repository_path="$2"
branch="$3"
trap 'rm -f "$bundle_path"' EXIT

if [[ -d "$repository_path/.git" ]]; then
  if ! git -C "$repository_path" diff --quiet || \
    ! git -C "$repository_path" diff --cached --quiet; then
    echo "Refusing to replace modified tracked files in $repository_path." >&2
    exit 1
  fi
  git -C "$repository_path" fetch "$bundle_path" "refs/heads/$branch"
  git -C "$repository_path" checkout "$branch"
  git -C "$repository_path" merge --ff-only FETCH_HEAD
else
  mkdir -p "$(dirname "$repository_path")"
  git clone --branch "$branch" "$bundle_path" "$repository_path"
fi

cd "$repository_path"
supervisor_status="$(supervisorctl status voice-light-compute 2>/dev/null || true)"
if [[ "$supervisor_status" == voice-light-compute* ]]; then
  supervisorctl stop voice-light-compute
fi
bash deployment/compute/bootstrap.sh
bash deployment/compute/install-service.sh
