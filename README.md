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
uv run python -m app.server
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
uv run python -m app.db.migrate
```

Batch ASR, dataset quality analysis, and the voice prototype require the compute backend:

```text
VOICE_LIGHT_COMPUTE_URL=http://<vast-ip>:8000
VOICE_LIGHT_COMPUTE_TOKEN=<token from the compute .env.compute file>
```

The compute URL has no implicit deployment default. The local application fails clearly when a
compute-backed operation is requested without these values.

## Vast.ai compute backend

On a newly rented Ubuntu RTX 4090 instance:

```bash
git clone <repository-url>
cd Voice-Light
bash deployment/compute/bootstrap.sh
bash deployment/compute/start.sh
```

`bootstrap.sh` installs Linux audio/compiler packages, installs uv, synchronizes the locked Python
3.12 environment, validates the RTX 4090/CUDA runtime, caches required voice models, and performs
import and Pocket TTS streaming smoke tests. It creates an ignored `.env.compute` containing a new
bearer token. Copy the token securely into `VOICE_LIGHT_COMPUTE_TOKEN` on the local machine.

After a later pull, restart with:

```bash
git pull
bash deployment/compute/start.sh
```

The start command synchronizes only changed dependencies, stops the tracked process, and starts one
server on the configured port. Operational commands are:

```bash
bash deployment/compute/status.sh
bash deployment/compute/stop.sh
.venv/bin/python deployment/compute/benchmark_tts.py
```

See [provider-neutral compute backend](docs/compute-backend.md) for the deployment boundary,
endpoints, authentication, readiness behavior, and TTS decision.
