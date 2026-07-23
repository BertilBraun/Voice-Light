#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: provision-vast.sh <bundle-path> <repository-path> <branch> <full|asr>" >&2
  exit 1
fi

bundle_path="$1"
repository_path="$2"
branch="$3"
deployment_mode="$4"
if [[ "$deployment_mode" != "full" && "$deployment_mode" != "asr" ]]; then
  echo "Deployment mode must be either 'full' or 'asr'." >&2
  exit 1
fi
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
bash deployment/compute/bootstrap.sh --mode "$deployment_mode"
bash deployment/compute/install-service.sh
