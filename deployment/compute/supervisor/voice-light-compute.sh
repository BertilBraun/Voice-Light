#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

repository_root="${WORKSPACE}/Voice-Light"
cd "$repository_root"

set -a
source .env.compute
set +a

export PYTHONUNBUFFERED=1
export VOICE_LIGHT_COMPUTE_HOST=127.0.0.1
exec .venv/bin/python -m app.compute.server
