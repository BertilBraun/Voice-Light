# Voice Light

Small local browser app for inspecting conversational speech data and analysis outputs.

## Data

Place the dataset under the repository root `data/` folder. The current app expects local dataset
sessions at:

```text
data/dataset_1/samples/
```

The root `data/` folder is intentionally ignored by Git.

## Run

### Docker Compose

Start Postgres, apply migrations, and run the app:

```powershell
docker compose up
```

This builds the shared application image, including the ML dependencies used by the compute code.
For a lightweight Windows restart of only Postgres and the local web application, use the
instructions under **Local on Windows** instead.

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
http://127.0.0.1:8000/future-work
```

See [ASR analysis](docs/asr-analysis.md) for the model, caching, and post-processing workflow.
See [turn-taking adapter training](docs/turn-taking-training.md) for the dataset contract, model
choice, training schedule, and runnable training prototype.
See the [synthetic tool-use generator](docs/tool-use-synthetic-generation.md), the
[tool-use LoRA runbook](docs/tool-use-lora-training.md), and the
[measured fine-tuning results](docs/tool-use-finetuning-results.md) for the complete natural
spoken tool-use experiment.

## Future work

The [Future Work](http://127.0.0.1:8000/future-work) page automatically lists and renders every
Markdown file in `docs/future-work/`. To add an idea, create a kebab-case `.md` file whose first
line is an H1 title and whose first paragraph is a short summary for the index card. No application
code or navigation update is required.

The Compose app mounts the repository `data/` directory at `/app/data` in the
container. For local dataset ingestion, use:

```text
/app/data/dataset_1/samples
```

Postgres data is stored in the `voice-light-postgres` Docker volume.

### Local on Windows

The project's uv resolution is currently limited to Linux x86_64, and the shared dependency list
contains Linux-oriented ML packages. Do not use `uv run` for routine Windows restarts because it
may try to synchronize those dependencies. Use the existing `.venv` directly.

Start only the local Postgres service and apply migrations:

```powershell
docker compose up -d postgres
.\.venv\Scripts\python.exe -m app.local.db.migrate
```

Start the FastAPI server from the repository root:

```powershell
$env:VOICE_LIGHT_HOST = '127.0.0.1'
$env:VOICE_LIGHT_PORT = '8000'
$env:VOICE_LIGHT_RELOAD = 'false'
.\.venv\Scripts\python.exe -m app.local.server
```

To keep it running in the background:

```powershell
$env:VOICE_LIGHT_HOST = '127.0.0.1'
$env:VOICE_LIGHT_PORT = '8000'
$env:VOICE_LIGHT_RELOAD = 'false'
$runtimeDirectory = (New-Item -ItemType Directory -Force '.\.runtime').FullName

Start-Process `
  -FilePath '.\.venv\Scripts\python.exe' `
  -ArgumentList '-m', 'app.local.server' `
  -WorkingDirectory (Get-Location) `
  -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $runtimeDirectory 'local-server.stdout.log') `
  -RedirectStandardError (Join-Path $runtimeDirectory 'local-server.stderr.log')
```

Then open:

```text
http://127.0.0.1:8000
```

Set `VOICE_LIGHT_PORT` to use a different port.

For DB-backed dataset pages outside Docker, the app defaults to the Postgres service exposed by
this repository's Compose configuration at
`postgresql://voice_light:voice_light@127.0.0.1:5432/voice_light`. Override
`VOICE_LIGHT_DATABASE_URL` only when using a different database.

### Local on Linux

The uv-managed path remains available on Linux x86_64:

```powershell
docker compose up -d postgres
uv run python -m app.local.db.migrate
uv run python -m app.local.server
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
3.12 environment with the `compute` dependency extra, validates the NVIDIA/CUDA runtime and at
least 10 GiB of GPU memory, caches
required voice models, and performs a streaming TTS smoke test. Moshi, NeMo, librosa, and
faster-whisper are compute-only dependencies and are not installed for the local app. The script
creates an ignored `.env.compute` containing a new bearer token and
`VOICE_LIGHT_TTS_BACKEND=kyutai`. Set that value to `voxtream` and rerun bootstrap to install the
pinned isolated VoXtream environment. Copy the token securely into `VOICE_LIGHT_COMPUTE_TOKEN` on
the local machine.

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
.venv/bin/python -m deployment.compute.benchmark_tts
```

The conversational Qwen adapter can also be built as a standalone BF16 safetensors checkpoint.
The merge command always uses the exact base-model and adapter revisions pinned by Voice Light:

```powershell
uv run python -m deployment.compute.merge_qwen_lora `
  --output-directory '.runtime/qwen3-1.7b-tool-use-merged' `
  --destination-repository-id 'BertilBraun/qwen3-1.7b-voice-light-tool-use-merged'
```

The output includes the tokenizer, a Hugging Face model card, and machine-readable merge
provenance. After uploading it, configure both
`VOICE_LIGHT_MERGED_LANGUAGE_MODEL_NAME=BertilBraun/qwen3-1.7b-voice-light-tool-use-merged` and
`VOICE_LIGHT_MERGED_LANGUAGE_MODEL_REVISION=<Hugging Face commit hash>` in `.env.compute`.
Voice Light then loads that pinned checkpoint in vLLM with dynamic LoRA disabled. If neither
variable is present, it continues to load the separately pinned base model and LoRA adapter.

See [provider-neutral compute backend](docs/compute-backend.md) for the deployment boundary,
endpoints, authentication, readiness behavior, and TTS decision.
See [Vast.ai deployment](docs/vast-deployment.md) for the rental requirements, one-command local
deployment, replacement procedure, and cleanup boundary.
