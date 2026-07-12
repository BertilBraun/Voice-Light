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

For DB-backed dataset pages outside Docker, set `VOICE_LIGHT_DATABASE_URL` and run:

```powershell
uv run python -m app.db.migrate
```

Dataset ingestion also requires the remote quality-analysis service:

```text
VOICE_LIGHT_REMOTE_QUALITY_ENDPOINT_URL
VOICE_LIGHT_REMOTE_QUALITY_API_KEY
```

Deploy the Modal quality endpoint with:

```powershell
uv run modal deploy .\app\quality\modal_endpoint.py
```

Local audio is staged in a temporary request directory on a Modal Volume and deleted after the
request. Remote storage backends can provide an HTTP or HTTPS access URI, including a presigned
S3 URL, which Modal downloads into temporary storage for the duration of the analysis.
