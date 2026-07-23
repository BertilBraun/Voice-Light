#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

if [[ ! -f .env.compute ]]; then
  echo ".env.compute is missing. Run bash deployment/compute/bootstrap.sh first." >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is missing. Run bash deployment/compute/bootstrap.sh first." >&2
  exit 1
fi

set -a
source .env.compute
set +a

if [[ "${VOICE_LIGHT_VOICE_STACK_ENABLED:-true}" == "false" ]]; then
  uv sync --frozen --python 3.12 --extra compute \
    --no-install-package moshi --no-install-package peft
  bash deployment/compute/install_asr_torch_cu126.sh
else
  uv sync --frozen --python 3.12 --extra compute
fi

supervisor_status=""
if command -v supervisorctl >/dev/null 2>&1; then
  supervisor_status="$(supervisorctl status voice-light-compute 2>/dev/null || true)"
fi
if [[ "$supervisor_status" == voice-light-compute* ]]; then
  supervisorctl restart voice-light-compute
  echo "Compute server restarted under Supervisor."
  exit 0
fi

bash deployment/compute/stop.sh

port="${VOICE_LIGHT_COMPUTE_PORT:-8000}"
if ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
  echo "Port $port is already in use by an untracked process. Stop it before starting Voice Light." >&2
  exit 1
fi

mkdir -p logs/compute run/compute
nohup .venv/bin/python -m app.compute.server >> logs/compute/server.log 2>&1 &
server_pid=$!
echo "$server_pid" > run/compute/server.pid

for _attempt in {1..30}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Compute server exited during startup. See logs/compute/server.log." >&2
    exit 1
  fi
  if curl --silent --fail "http://127.0.0.1:$port/health/live" >/dev/null; then
    echo "Compute server started with PID $server_pid on port $port."
    echo "Model loading continues in the background; use bash deployment/compute/status.sh."
    exit 0
  fi
  sleep 1
done

echo "Compute server did not become live within 30 seconds. See logs/compute/server.log." >&2
exit 1
