# Voice Light

Small local browser app for inspecting conversational speech data and analysis outputs.

## Data

Place the dataset under the repository root `data/` folder. The current app expects LUEL
sessions at:

```text
data/luel/sessions/
```

The root `data/` folder is intentionally ignored by Git.

## Run

### Docker Compose

Start Postgres, apply migrations, and run the app:

```powershell
docker compose up
```

Then open:

```text
http://127.0.0.1:8000
```

Useful pages:

```text
http://127.0.0.1:8000/datasets
http://127.0.0.1:8000/datasets/ingest
http://127.0.0.1:8000/analyses/end-of-turn
http://127.0.0.1:8000/analyses/asr
```

See [ASR analysis](docs/asr-analysis.md) for the model, caching, and post-processing workflow.
See [turn-taking adapter training](docs/turn-taking-training.md) for the dataset contract, model
choice, training schedule, and runnable training prototype.

The Compose app mounts the repository `data/` directory at `/app/data` in the
container. For local LUEL ingestion, use:

```text
/app/data/luel/sessions
```

Postgres data is stored in the `voice-light-postgres` Docker volume.

### Local

Start the FastAPI server from the repository root:

```powershell
uv run python -m app.local.server
```

Then open:

```text
http://127.0.0.1:8000
```

Set `VOICE_LIGHT_PORT` to use a different port.

For DB-backed dataset pages outside Docker, the app defaults to the Postgres service exposed by
this repository's Compose configuration at
`postgresql://voice_light:voice_light@127.0.0.1:5432/voice_light`. Override
`VOICE_LIGHT_DATABASE_URL` only when using a different database. Run migrations with:

```powershell
uv run python -m app.local.db.migrate
```

Batch ASR and dataset quality analysis require the local application to know the compute backend:

```text
VOICE_LIGHT_COMPUTE_URL=http://<vast-ip>:8000
VOICE_LIGHT_COMPUTE_TOKEN=<token from the compute .env.compute file>
```

The compute URL has no implicit deployment default. The local application fails clearly when a
compute-backed operation is requested without these values.

The voice prototype instead connects the browser directly to the compute service. Open
`http://127.0.0.1:8000/voice-agent` and enter `ws://<vast-ip>:8000/v1/voice` in the endpoint field.
The ephemeral research WebSocket does not use the HTTP bearer token.

## Vast.ai compute backend

For a new Vast.ai PyTorch rental, deploy the committed local revision and open the private tunnel
with one PowerShell command:

```powershell
.\deployment\compute\deploy-vast.ps1 `
  -SshHost '<vast-ssh-host>' `
  -SshPort <vast-ssh-port> `
  -SshKeyPath "$HOME\.ssh\codex_vast_ed25519"
```

The command transfers the repository without requiring remote Git credentials, bootstraps the
locked compute environment, downloads and validates the models, installs an automatically
restarting Supervisor service, saves the new token in `.runtime/compute.env`, and verifies the
backend through `http://127.0.0.1:8080`.

To bootstrap from a shell already open on the rental instead:

```bash
git clone <repository-url> /workspace/Voice-Light
cd Voice-Light
bash deployment/compute/bootstrap.sh
bash deployment/compute/install-service.sh
```

`bootstrap.sh` installs Linux audio/compiler packages, installs uv, synchronizes the locked Python
3.12 environment with the `compute` dependency extra, validates the RTX 4090/CUDA runtime, caches
required voice models, and performs import and Kyutai TTS streaming smoke tests. Moshi, NeMo,
librosa, and faster-whisper are compute-only dependencies and are not installed for the local app.
The script creates an ignored `.env.compute` containing a new bearer token. Copy the token securely
into `VOICE_LIGHT_COMPUTE_TOKEN` on the local machine.

After a later pull, synchronize dependencies and restart the Supervisor service with:

```bash
git pull
bash deployment/compute/start.sh
```

The start command synchronizes only changed dependencies and restarts the managed service.
Operational commands are:

```bash
bash deployment/compute/status.sh
bash deployment/compute/stop.sh
.venv/bin/python deployment/compute/benchmark_tts.py
```

See [provider-neutral compute backend](docs/compute-backend.md) for the deployment boundary,
endpoints, authentication, readiness behavior, and TTS decision.
See [Vast.ai deployment](docs/vast-deployment.md) for the rental requirements, one-command local
deployment, replacement procedure, and cleanup boundary.
