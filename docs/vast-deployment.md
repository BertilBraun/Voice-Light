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
2. Installs system packages, Python 3.12, and the locked compute dependencies.
3. Creates a fresh `.env.compute` token when the instance does not already have one.
4. Downloads and validates the required models and CUDA environment.
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

## Replace or destroy an instance

The compute application has no persistent database state. Before destroying an instance, preserve
only artifacts intentionally created there, such as benchmark output or training results. Model
caches and `.env.compute` are disposable.

For the replacement rental, inject the same SSH public key and run `deploy-vast.ps1` with its new
hostname and SSH port. The command replaces the old local tunnel and retrieves the replacement
token. Update the local application's `VOICE_LIGHT_COMPUTE_TOKEN` from
`.runtime/compute.env` if batch ASR or quality analysis is used. The browser voice endpoint remains
`ws://127.0.0.1:8080/v1/voice` when the default local port is retained.

Do not destroy the old rental until the replacement command reports that the local health endpoint
is working. This makes replacement reversible and keeps the interruption short.
