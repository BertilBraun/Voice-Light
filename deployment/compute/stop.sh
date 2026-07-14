#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

if command -v supervisorctl >/dev/null 2>&1 && \
  supervisorctl status voice-light-compute >/dev/null 2>&1; then
  supervisorctl stop voice-light-compute
  echo "Compute server stopped under Supervisor."
  exit 0
fi

pid_file="run/compute/server.pid"

if [[ ! -f "$pid_file" ]]; then
  echo "Compute server is not running."
  exit 0
fi

server_pid="$(<"$pid_file")"
if ! [[ "$server_pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid PID file: $pid_file" >&2
  exit 1
fi

if ! kill -0 "$server_pid" 2>/dev/null; then
  rm -f "$pid_file"
  echo "Removed stale compute server PID file."
  exit 0
fi

kill -TERM "$server_pid"
for _attempt in {1..30}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "Compute server stopped."
    exit 0
  fi
  sleep 1
done

kill -KILL "$server_pid"
rm -f "$pid_file"
echo "Compute server required SIGKILL after the 30-second graceful-shutdown window." >&2
