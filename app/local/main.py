from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from app.local.analyses.asr.router import router as asr_analysis_router
from app.local.analyses.end_of_turn.router import router as end_of_turn_router
from app.local.asr.router import router as asr_router
from app.local.config import COMPUTE_BASE_URL, COMPUTE_TOKEN, WEB_ROOT
from app.local.dashboard.router import router as dataset_dashboard_router
from app.local.data.sessions import SessionEntry, SpeakerName, list_sessions, session_audio_path
from app.local.voice.compute_client import (
    RemoteComputeVoiceChannel,
    RemoteLanguageModel,
    RemoteSpeechSynthesizer,
    RemoteStreamingTranscriber,
)
from app.local.voice.session import SessionPolicy, VoiceAgentSession, send_session_error
from app.local.voice.speech_detection import SileroSpeechDetector
from app.shared.audio.wav import capped_wave_bytes
from app.shared.base_model import FrozenBaseModel

SESSIONS_CACHE_CONTROL = "public, max-age=31536000, immutable"


class SessionListResponse(FrozenBaseModel):
    sessions: tuple[SessionEntry, ...]


app = FastAPI(title="Voice Light")
app.include_router(asr_router)
app.include_router(asr_analysis_router)
app.include_router(end_of_turn_router)
app.include_router(dataset_dashboard_router)
app.mount("/pages", StaticFiles(directory=WEB_ROOT / "pages"), name="pages")


@app.get("/")
def overview_page() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/analyses/end-of-turn")
def end_of_turn_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "end-of-turn" / "index.html", headers={"Cache-Control": "no-store"}
    )


@app.get("/analyses/asr")
def asr_analysis_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "asr" / "index.html", headers={"Cache-Control": "no-store"}
    )


@app.get("/voice-agent")
def voice_agent_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "voice-agent" / "index.html", headers={"Cache-Control": "no-store"}
    )


@app.websocket("/api/voice/session")
async def voice_agent_session(websocket: WebSocket) -> None:
    channel = RemoteComputeVoiceChannel(base_url=COMPUTE_BASE_URL, token=COMPUTE_TOKEN)
    try:
        await channel.connect()
    except Exception as error:
        await websocket.accept()
        await send_session_error(websocket=websocket, error=error)
        await websocket.close(code=1013)
        return
    try:
        session = VoiceAgentSession(
            websocket=websocket,
            speech_detector=SileroSpeechDetector(),
            transcriber=RemoteStreamingTranscriber(channel),
            language_model=RemoteLanguageModel(channel),
            speech_synthesizer=RemoteSpeechSynthesizer(channel),
            policy=SessionPolicy(),
        )
        await session.run()
    except Exception as error:
        await send_session_error(websocket=websocket, error=error)
        await websocket.close(code=1011)
    finally:
        await channel.close()


@app.get("/datasets")
def dataset_dashboard_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "datasets" / "index.html", headers={"Cache-Control": "no-store"}
    )


@app.get("/datasets/ingest")
def dataset_ingestion_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "datasets" / "ingest.html", headers={"Cache-Control": "no-store"}
    )


@app.get("/api/sessions")
def sessions_api(response: Response) -> SessionListResponse:
    response.headers["Cache-Control"] = SESSIONS_CACHE_CONTROL
    return SessionListResponse(sessions=list_sessions())


@app.get("/api/audio/{identifier}/{speaker_name}")
def audio_api(identifier: str, speaker_name: SpeakerName) -> FileResponse:
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
        prefix=f"{wave_path.stem}_playback_", suffix=".wav", delete=False
    ) as playback_wave_file:
        playback_wave_file.write(capped_wave_bytes(wave_path=wave_path))
        return Path(playback_wave_file.name)


def delete_file(path: Path) -> None:
    path.unlink(missing_ok=True)
