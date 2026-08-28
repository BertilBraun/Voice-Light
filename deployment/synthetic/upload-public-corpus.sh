#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
staging_root="${VOICE_LIGHT_SYNTHETIC_STAGING_ROOT:-/workspace/voice-light-synthetic-publication-v3}"
repository_id="${VOICE_LIGHT_SYNTHETIC_REPOSITORY_ID:-BertilBraun/voice-light-synthetic-audio}"

exec /venv/main/bin/hf upload-large-folder \
  "$repository_id" \
  "$staging_root" \
  --repo-type dataset \
  --include README.md \
  --include 'builder/*' \
  --include 'runs/v4/**' \
  --include 'runs/v5/**' \
  --num-workers 8 \
  --no-bars
