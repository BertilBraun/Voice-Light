from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from app.analyses.asr.router import router as asr_router
from app.analyses.end_of_turn.router import router as end_of_turn_router
from app.audio.wav import capped_wave_bytes
from app.config import WEB_ROOT
from app.data.sessions import SpeakerName, list_sessions, session_audio_path, session_to_json
from app.dataset_dashboard.router import router as dataset_dashboard_router

app = FastAPI(title="Voice Light")
app.include_router(asr_router)
app.include_router(end_of_turn_router)
app.include_router(dataset_dashboard_router)
app.mount("/pages", StaticFiles(directory=WEB_ROOT / "pages"), name="pages")


@app.get("/")
def overview_page() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/analyses/end-of-turn")
def end_of_turn_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "end-of-turn" / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/analyses/asr")
def asr_analysis_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "asr" / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/datasets")
def dataset_dashboard_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "datasets" / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/datasets/ingest")
def dataset_ingestion_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "datasets" / "ingest.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/sessions")
def sessions_api() -> dict[str, object]:
    return {"sessions": [session_to_json(session_entry) for session_entry in list_sessions()]}


@app.get("/api/audio/{identifier}/{speaker_name}")
def audio_api(
    identifier: str,
    speaker_name: SpeakerName,
) -> FileResponse:
    try:
        wave_path = session_audio_path(identifier=identifier, speaker_name=speaker_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    playback_wave_path = write_playback_wave_file(wave_path=wave_path)
    return FileResponse(
        playback_wave_path,
        media_type="audio/wav",
        filename=playback_wave_path.name,
        headers={"Cache-Control": "no-store"},
        background=BackgroundTask(delete_file, path=playback_wave_path),
    )


def write_playback_wave_file(wave_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"{wave_path.stem}_playback_",
        suffix=".wav",
        delete=False,
    ) as playback_wave_file:
        playback_wave_file.write(capped_wave_bytes(wave_path=wave_path))
        return Path(playback_wave_file.name)


def delete_file(path: Path) -> None:
    path.unlink(missing_ok=True)
