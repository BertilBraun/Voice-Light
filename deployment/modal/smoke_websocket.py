from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

from websockets.asyncio.client import connect

from app.compute.voice.schemas import SessionReadyEvent, SessionStartEvent

DEFAULT_WEBSOCKET_URL = (
    "wss://bertil-braun-private--voicelightagent-voice-light.eu-west.modal.run/v1/voice"
)
DEFAULT_OPEN_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class SmokeResult:
    session_id: str
    ready_seconds: float


async def run_smoke(websocket_url: str, open_timeout_seconds: float) -> SmokeResult:
    if open_timeout_seconds <= 0:
        raise ValueError("WebSocket open timeout must be positive.")
    started = time.perf_counter()
    async with connect(websocket_url, open_timeout=open_timeout_seconds) as websocket:
        await websocket.send(
            SessionStartEvent(
                input_sample_rate=16_000,
                local_time_zone="Europe/Berlin",
            ).model_dump_json()
        )
        message = await asyncio.wait_for(websocket.recv(), timeout=30)
    match message:
        case str():
            ready = SessionReadyEvent.model_validate_json(message)
        case bytes():
            raise RuntimeError("Expected a session.ready JSON event, received binary audio.")
    return SmokeResult(
        session_id=ready.session_id,
        ready_seconds=time.perf_counter() - started,
    )


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_WEBSOCKET_URL)
    parser.add_argument(
        "--open-timeout-seconds",
        default=DEFAULT_OPEN_TIMEOUT_SECONDS,
        type=float,
    )
    options = parser.parse_args(arguments)
    result = asyncio.run(run_smoke(options.url, options.open_timeout_seconds))
    print(f"session_id={result.session_id} ready_seconds={result.ready_seconds:.3f}")


if __name__ == "__main__":
    main()
