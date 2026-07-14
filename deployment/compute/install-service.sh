#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "install-service.sh must run as root." >&2
  exit 1
fi

if ! command -v supervisorctl >/dev/null 2>&1; then
  echo "supervisorctl is missing. Use a Vast.ai base image with Supervisor." >&2
  exit 1
fi

install -m 755 \
  "$repository_root/deployment/compute/supervisor/voice-light-compute.sh" \
  /opt/supervisor-scripts/voice-light-compute.sh
install -m 644 \
  "$repository_root/deployment/compute/supervisor/voice-light-compute.conf" \
  /etc/supervisor/conf.d/voice-light-compute.conf

service_was_installed=false
if supervisorctl status voice-light-compute >/dev/null 2>&1; then
  service_was_installed=true
fi
supervisorctl reread
supervisorctl update
if [[ "$service_was_installed" == true ]]; then
  supervisorctl restart voice-light-compute
fi

for _attempt in {1..60}; do
  if curl --silent --fail http://127.0.0.1:8000/health/live >/dev/null; then
    echo "Voice Light compute is managed by Supervisor and is live on port 8000."
    exit 0
  fi
  sleep 1
done

echo "Voice Light compute did not become live within 60 seconds." >&2
echo "Inspect /var/log/portal/voice-light-compute.log on the instance." >&2
exit 1
