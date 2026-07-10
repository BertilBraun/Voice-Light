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

Start the FastAPI server from the repository root:

```powershell
uv run python -m app.server
```

Then open:

```text
http://127.0.0.1:8765
```

Set `VOICE_LIGHT_PORT` to use a different port.
