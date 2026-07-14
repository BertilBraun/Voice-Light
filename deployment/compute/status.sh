#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

if [[ ! -f .env.compute ]]; then
  echo ".env.compute is missing. Run bootstrap.sh first." >&2
  exit 1
fi

set -a
source .env.compute
set +a

port="${VOICE_LIGHT_COMPUTE_PORT:-8000}"
if command -v supervisorctl >/dev/null 2>&1 && \
  supervisorctl status voice-light-compute >/dev/null 2>&1; then
  supervisorctl status voice-light-compute
  curl --silent --show-error "http://127.0.0.1:$port/health/live"
  echo
  curl --silent --show-error \
    --header "Authorization: Bearer $VOICE_LIGHT_COMPUTE_TOKEN" \
    "http://127.0.0.1:$port/health/ready"
  echo
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  exit 0
fi

pid_file="run/compute/server.pid"
if [[ ! -f "$pid_file" ]] || ! kill -0 "$(<"$pid_file")" 2>/dev/null; then
  echo "Compute server: stopped"
  exit 1
fi

echo "Compute server: running (PID $(<"$pid_file"))"
curl --silent --show-error "http://127.0.0.1:$port/health/live"
echo
curl --silent --show-error \
  --header "Authorization: Bearer $VOICE_LIGHT_COMPUTE_TOKEN" \
  "http://127.0.0.1:$port/health/ready"
echo
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
