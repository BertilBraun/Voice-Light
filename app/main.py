from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.analyses.end_of_turn.router import router as end_of_turn_router
from app.audio.wav import capped_wave_bytes
from app.config import WEB_ROOT
from app.data.sessions import SpeakerName, list_sessions, session_audio_path, session_to_json

app = FastAPI(title="Voice Light")
app.include_router(end_of_turn_router)
app.mount("/pages", StaticFiles(directory=WEB_ROOT / "pages"), name="pages")

BYTE_RANGE_UNIT = "bytes"


@dataclass(frozen=True)
class ByteRange:
    start_index: int
    end_index: int


@app.get("/")
def overview_page() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/analyses/end-of-turn")
def end_of_turn_page() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "pages" / "end-of-turn" / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/sessions")
def sessions_api() -> dict[str, object]:
    return {"sessions": [session_to_json(session_entry) for session_entry in list_sessions()]}


@app.get("/api/audio/{identifier}/{speaker_name}")
def audio_api(
    identifier: str,
    speaker_name: SpeakerName,
    range_header: str | None = Header(default=None, alias="Range"),
) -> Response:
    try:
        wave_path = session_audio_path(identifier=identifier, speaker_name=speaker_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    wave_bytes = capped_wave_bytes(wave_path=wave_path)
    base_headers = {
        "Accept-Ranges": BYTE_RANGE_UNIT,
        "Cache-Control": "no-store",
    }
    if range_header is None:
        return Response(
            content=wave_bytes,
            headers={**base_headers, "Content-Length": str(len(wave_bytes))},
            media_type="audio/wav",
        )

    try:
        byte_range = parse_byte_range(range_header=range_header, content_length=len(wave_bytes))
    except ValueError as error:
        raise HTTPException(
            status_code=416,
            detail=str(error),
            headers={"Content-Range": f"{BYTE_RANGE_UNIT} */{len(wave_bytes)}"},
        ) from error

    partial_content = wave_bytes[byte_range.start_index : byte_range.end_index + 1]
    return Response(
        content=partial_content,
        headers={
            **base_headers,
            "Content-Length": str(len(partial_content)),
            "Content-Range": (
                f"{BYTE_RANGE_UNIT} {byte_range.start_index}-{byte_range.end_index}/"
                f"{len(wave_bytes)}"
            ),
        },
        media_type="audio/wav",
        status_code=206,
    )


def parse_byte_range(range_header: str, content_length: int) -> ByteRange:
    if content_length <= 0:
        raise ValueError("Cannot serve a byte range for empty content.")
    unit, separator, range_value = range_header.partition("=")
    if unit.strip() != BYTE_RANGE_UNIT or separator != "=":
        raise ValueError(f"Unsupported Range header: {range_header}")
    start_value, dash, end_value = range_value.partition("-")
    if dash != "-" or "," in range_value:
        raise ValueError(f"Unsupported Range header: {range_header}")

    if start_value == "":
        return suffix_byte_range(end_value=end_value, content_length=content_length)
    return bounded_byte_range(
        start_value=start_value,
        end_value=end_value,
        content_length=content_length,
    )


def suffix_byte_range(end_value: str, content_length: int) -> ByteRange:
    if end_value == "":
        raise ValueError("Range header is missing a byte count.")
    suffix_length = int(end_value)
    if suffix_length <= 0:
        raise ValueError("Range suffix length must be positive.")
    start_index = max(0, content_length - suffix_length)
    return ByteRange(start_index=start_index, end_index=content_length - 1)


def bounded_byte_range(
    start_value: str,
    end_value: str,
    content_length: int,
) -> ByteRange:
    start_index = int(start_value)
    end_index = content_length - 1 if end_value == "" else int(end_value)
    if start_index < 0 or start_index >= content_length or end_index < start_index:
        raise ValueError("Requested byte range is not satisfiable.")
    return ByteRange(
        start_index=start_index,
        end_index=min(end_index, content_length - 1),
    )
