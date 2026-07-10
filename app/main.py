from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.analyses.end_of_turn.router import router as end_of_turn_router
from app.config import WEB_ROOT
from app.data.sessions import SpeakerName, list_sessions, session_audio_path, session_to_json

app = FastAPI(title="Voice Light")
app.include_router(end_of_turn_router)
app.mount("/pages", StaticFiles(directory=WEB_ROOT / "pages"), name="pages")


@app.get("/")
def overview_page() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/analyses/end-of-turn")
def end_of_turn_page() -> FileResponse:
    return FileResponse(WEB_ROOT / "pages" / "end-of-turn" / "index.html")


@app.get("/api/sessions")
def sessions_api() -> dict[str, object]:
    return {"sessions": [session_to_json(session_entry) for session_entry in list_sessions()]}


@app.get("/api/audio/{identifier}/{speaker_name}")
def audio_api(identifier: str, speaker_name: SpeakerName) -> FileResponse:
    try:
        wave_path = session_audio_path(identifier=identifier, speaker_name=speaker_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return FileResponse(wave_path, media_type="audio/wav")
