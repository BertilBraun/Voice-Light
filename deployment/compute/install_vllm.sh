#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_project="$repository_root/deployment/compute/vllm"

uv sync \
  --project "$runtime_project" \
  --frozen \
  --python 3.12

echo "vLLM environment installed at $runtime_project/.venv."
