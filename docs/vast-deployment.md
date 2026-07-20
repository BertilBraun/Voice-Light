# Vast.ai deployment

Voice Light compute instances are disposable. A new rental needs an NVIDIA GPU, the Vast.ai
PyTorch base image, sufficient disk for the locked environment and model caches, an injected SSH
public key, and an SSH port. No public application port is required because the local workflow
uses an SSH tunnel.

## Deploy a new rental

Wait until Vast.ai shows the instance as running, then copy the SSH hostname, SSH port, and private
key path into this PowerShell command from the local repository root:

```powershell
.\deployment\compute\deploy-vast.ps1 `
  -SshHost '203.0.113.10' `
  -SshPort 42022 `
  -SshKeyPath "$HOME\.ssh\codex_vast_ed25519"
```

The deployment command refuses uncommitted tracked changes. It creates a Git bundle from the
current branch, transfers that exact revision without requiring Git credentials on the rental,
and then performs the following operations:

1. Clones or fast-forwards `/workspace/Voice-Light`.
2. Installs system packages, Python 3.12, the locked compute dependencies, and the isolated locked
   vLLM environment.
3. Creates a fresh `.env.compute` token when the instance does not already have one.
4. Downloads and validates the required models, the pinned conversational LoRA adapter, and the
   CUDA environment.
5. Installs the compute server as a Supervisor service with automatic restart.
6. Copies the compute environment to the ignored local `.runtime/compute.env` file.
7. Replaces the tracked SSH tunnel and verifies the service through `127.0.0.1:8080`.

Bootstrap time is dominated by dependency and model downloads. Re-running the command against the
same instance reuses its caches and preserves its existing token. A completely new rental gets a
new token and fresh caches.

Use a different local port when 8080 is needed by another application:

```powershell
.\deployment\compute\deploy-vast.ps1 -SshHost '203.0.113.10' -SshPort 42022 -LocalPort 8081
```

## Verify and operate the remote service

```bash
cd /workspace/Voice-Light
bash deployment/compute/status.sh
supervisorctl status voice-light-compute
tail -f /var/log/portal/voice-light-compute.log
```

After deploying a later commit with `deploy-vast.ps1`, the service is restarted automatically.
When updating directly on the instance, use:

```bash
cd /workspace/Voice-Light
bash deployment/compute/start.sh
```

## Verify the conversational LoRA

Bootstrap prints a cache line containing the adapter repository and immutable revision. After the
service starts, `/var/log/portal/voice-light-compute.log` contains an `activated Qwen adapter`
record. The conversational vLLM child process command also includes both `--adapter` and
`--adapter-revision`; the search summarizer has neither argument.

Run the behavioral smoke test only in an exclusive window. It loads the same Qwen base and adapter
as production, so first stop the Supervisor service to avoid loading a second copy. The shell trap
restores the single service even when a smoke case fails:

```bash
cd /workspace/Voice-Light
supervisorctl status voice-light-compute
supervisorctl stop voice-light-compute
restart_voice_light() { supervisorctl start voice-light-compute; }
trap restart_voice_light EXIT
PYTHONPATH=/workspace/Voice-Light \
  .venv/bin/python -m deployment.compute.smoke_test_tool_use
supervisorctl start voice-light-compute
trap - EXIT
bash deployment/compute/status.sh
```

The JSON report covers an ordinary no-tool response, a schema-valid `calculate` call, a
schema-valid `search` call, and a nonempty post-tool continuation without another call. Each
tool-call case also requires audible bridge text and passes the production Hermes parser. This
smoke test validates model behavior and call structure without executing live search.

## Replace or destroy an instance

The compute application has no persistent database state. Before destroying an instance, preserve
only artifacts intentionally created there, such as benchmark output or training results. Model
caches and `.env.compute` are disposable.

For the replacement rental, inject the same SSH public key and run `deploy-vast.ps1` with its new
hostname and SSH port. The command replaces the old local tunnel and retrieves the replacement
token. Update the local application's `VOICE_LIGHT_COMPUTE_TOKEN` from
`.runtime/compute.env` if batch ASR or quality analysis is used. The browser voice endpoint remains
`ws://127.0.0.1:8080/v1/voice` when the default local port is retained.

For S3 manifest ingestion, also configure the compute instance's `.env.compute` with the standard
AWS credential and region variables. Cached source audio remains under
`VOICE_LIGHT_DATASET_AUDIO_CACHE_DIR` for the life of the instance. The local dashboard downloads
only audio opened for listening or waveform/sample inspection and retains it under
`VOICE_LIGHT_LOCAL_DATASET_AUDIO_CACHE_DIR` (default `.cache/local/dataset-audio`). Configure the
same standard AWS variables for the local app when those previews require authenticated access.

Do not destroy the old rental until the replacement command reports that the local health endpoint
is working. This makes replacement reversible and keeps the interruption short.
