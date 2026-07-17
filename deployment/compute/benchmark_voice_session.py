from __future__ import annotations

import argparse
import asyncio
import math
import struct
import sys
import time
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, TypeAdapter
from websockets.asyncio.client import ClientConnection, connect

from app.compute.voice.schemas import (
    AssistantAudioBoundaryEvent,
    AssistantTextDeltaEvent,
    ErrorEvent,
    LlmHistoryEvent,
    PlaybackCompleteEvent,
    PlaybackStartedEvent,
    SessionReadyEvent,
    SessionStartEvent,
    SessionStopEvent,
    SpeechStateEvent,
    TranscriptEvent,
    VoiceServerEvent,
    VoiceServerEventType,
)
from app.shared.base_model import FrozenBaseModel

INPUT_SAMPLE_RATE = 16_000
PCM_BYTES_PER_SAMPLE = 2
PCM_FRAME_DURATION_MS = 20
PCM_FRAME_BYTES = INPUT_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE * PCM_FRAME_DURATION_MS // 1_000
VOICE_SERVER_EVENT_ADAPTER: TypeAdapter[VoiceServerEvent] = TypeAdapter(VoiceServerEvent)


class VoiceTrialResult(FrozenBaseModel):
    trial: int = Field(gt=0)
    generation_id: int = Field(gt=0)
    recognized_text: str
    session_ready_ms: float = Field(ge=0.0)
    true_end_to_vad_endpoint_ms: float = Field(ge=0.0)
    true_end_to_commit_ms: float = Field(ge=0.0)
    commit_to_first_pcm_ms: float = Field(ge=0.0)
    true_end_to_first_pcm_ms: float = Field(ge=0.0)
    response_complete_ms: float = Field(ge=0.0)
    speculative_output_before_commit: bool


class VoiceBenchmarkReport(FrozenBaseModel):
    url: str
    wav_path: str
    trials: tuple[VoiceTrialResult, ...]
    commit_to_first_pcm_p50_ms: float
    commit_to_first_pcm_p90_ms: float
    commit_to_first_pcm_p95_ms: float
    true_end_to_first_pcm_p50_ms: float


@dataclass
class AudioSendTiming:
    true_end_at: float | None = None


@dataclass
class TrialObservations:
    ready_at: float | None = None
    vad_endpoint_at: float | None = None
    committed_at: float | None = None
    first_pcm_at: float | None = None
    response_complete_at: float | None = None
    generation_id: int | None = None
    recognized_text: str = ""
    speculative_output_before_commit: bool = False


def read_pcm16_mono(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as audio_file:
        if audio_file.getframerate() != INPUT_SAMPLE_RATE:
            raise ValueError(f"Benchmark audio must use {INPUT_SAMPLE_RATE} Hz.")
        if audio_file.getnchannels() != 1:
            raise ValueError("Benchmark audio must be mono.")
        if audio_file.getsampwidth() != PCM_BYTES_PER_SAMPLE:
            raise ValueError("Benchmark audio must use PCM16 samples.")
        return audio_file.readframes(audio_file.getnframes())


async def send_audio(
    websocket: ClientConnection,
    speech_pcm: bytes,
    tail_silence_ms: int,
    timing: AudioSendTiming,
) -> None:
    speech_started_at = time.perf_counter()
    for start in range(0, len(speech_pcm), PCM_FRAME_BYTES):
        frame = speech_pcm[start : start + PCM_FRAME_BYTES]
        await websocket.send(frame)
        await asyncio.sleep(len(frame) / (INPUT_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE))
    timing.true_end_at = speech_started_at + len(speech_pcm) / (
        INPUT_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE
    )
    silence_frame = bytes(PCM_FRAME_BYTES)
    for _ in range(math.ceil(tail_silence_ms / PCM_FRAME_DURATION_MS)):
        await websocket.send(silence_frame)
        await asyncio.sleep(PCM_FRAME_DURATION_MS / 1_000)


async def run_trial(
    url: str,
    speech_pcm: bytes,
    tail_silence_ms: int,
    trial: int,
    trace_transcripts: bool,
) -> VoiceTrialResult:
    connected_at = time.perf_counter()
    observations = TrialObservations()
    send_timing = AudioSendTiming()
    async with connect(url, max_size=None) as websocket:
        await websocket.send(
            SessionStartEvent(input_sample_rate=INPUT_SAMPLE_RATE).model_dump_json()
        )
        audio_task: asyncio.Task[None] | None = None
        async for message in websocket:
            now = time.perf_counter()
            match message:
                case bytes():
                    if observations.committed_at is None:
                        observations.speculative_output_before_commit = True
                    generation_id, _, _ = parse_audio_header(message)
                    if observations.first_pcm_at is None:
                        observations.first_pcm_at = now
                        observations.generation_id = generation_id
                        await websocket.send(
                            PlaybackStartedEvent(
                                generation_id=generation_id,
                                browser_monotonic_time_ns=time.perf_counter_ns(),
                                rendered_output_sample_position=1,
                                source_sample_position=1,
                                output_sample_rate=24_000,
                            ).model_dump_json()
                        )
                case str():
                    event = VOICE_SERVER_EVENT_ADAPTER.validate_json(message)
                    if trace_transcripts:
                        match event:
                            case TranscriptEvent():
                                observed_at = datetime.now(UTC).isoformat(timespec="milliseconds")
                                print(
                                    f"TRANSCRIPT_TRACE trial={trial} "
                                    f"observed_at={observed_at} "
                                    f"type={event.type} text={event.text!r}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            case _:
                                pass
                    match event:
                        case SessionReadyEvent():
                            observations.ready_at = now
                            audio_task = asyncio.create_task(
                                send_audio(
                                    websocket=websocket,
                                    speech_pcm=speech_pcm,
                                    tail_silence_ms=tail_silence_ms,
                                    timing=send_timing,
                                )
                            )
                        case SpeechStateEvent(type=VoiceServerEventType.VAD_STOPPED):
                            if observations.vad_endpoint_at is None:
                                observations.vad_endpoint_at = now
                        case TranscriptEvent(type=VoiceServerEventType.TURN_COMMITTED):
                            observations.committed_at = now
                            observations.recognized_text = event.text
                        case TranscriptEvent(
                            type=VoiceServerEventType.TRANSCRIPT_FINAL,
                            text="",
                        ):
                            await websocket.send(SessionStopEvent().model_dump_json())
                            raise RuntimeError(
                                "The benchmark utterance produced an empty final transcript."
                            )
                        case AssistantTextDeltaEvent():
                            if observations.committed_at is None:
                                observations.speculative_output_before_commit = True
                        case LlmHistoryEvent():
                            observations.generation_id = event.generation_id
                        case AssistantAudioBoundaryEvent(
                            type=VoiceServerEventType.ASSISTANT_AUDIO_END
                        ):
                            observations.response_complete_at = now
                            observations.generation_id = event.generation_id
                            await websocket.send(
                                PlaybackCompleteEvent(
                                    generation_id=event.generation_id,
                                    browser_monotonic_time_ns=time.perf_counter_ns(),
                                    rendered_output_sample_position=1,
                                    source_sample_position=1,
                                    output_sample_rate=24_000,
                                ).model_dump_json()
                            )
                            await websocket.send(SessionStopEvent().model_dump_json())
                            break
                        case ErrorEvent():
                            raise RuntimeError(
                                f"{event.component}/{event.operation}: {event.message}"
                            )
                        case _:
                            pass
        if audio_task is not None:
            await audio_task
    return build_trial_result(
        trial=trial,
        connected_at=connected_at,
        send_timing=send_timing,
        observations=observations,
    )


def parse_audio_header(message: bytes) -> tuple[int, int, int]:
    if len(message) < 12:
        raise ValueError("Voice audio frames must include the 12-byte media header.")
    return struct.unpack("<III", message[:12])


def build_trial_result(
    trial: int,
    connected_at: float,
    send_timing: AudioSendTiming,
    observations: TrialObservations,
) -> VoiceTrialResult:
    true_end_at = require_timestamp(send_timing.true_end_at, "true speech end")
    ready_at = require_timestamp(observations.ready_at, "session readiness")
    vad_endpoint_at = require_timestamp(observations.vad_endpoint_at, "VAD endpoint")
    committed_at = require_timestamp(observations.committed_at, "turn commitment")
    first_pcm_at = require_timestamp(observations.first_pcm_at, "first PCM")
    response_complete_at = require_timestamp(
        observations.response_complete_at,
        "response completion",
    )
    generation_id = observations.generation_id
    if generation_id is None:
        raise RuntimeError("The voice trial received no generation identifier.")
    return VoiceTrialResult(
        trial=trial,
        generation_id=generation_id,
        recognized_text=observations.recognized_text,
        session_ready_ms=milliseconds_between(connected_at, ready_at),
        true_end_to_vad_endpoint_ms=milliseconds_between(true_end_at, vad_endpoint_at),
        true_end_to_commit_ms=milliseconds_between(true_end_at, committed_at),
        commit_to_first_pcm_ms=milliseconds_between(committed_at, first_pcm_at),
        true_end_to_first_pcm_ms=milliseconds_between(true_end_at, first_pcm_at),
        response_complete_ms=milliseconds_between(committed_at, response_complete_at),
        speculative_output_before_commit=observations.speculative_output_before_commit,
    )


async def run_benchmark(
    url: str,
    wav_path: Path,
    trial_count: int,
    tail_silence_ms: int,
    trace_transcripts: bool,
) -> VoiceBenchmarkReport:
    speech_pcm = read_pcm16_mono(wav_path)
    trials = tuple(
        [
            await run_trial(
                url=url,
                speech_pcm=speech_pcm,
                tail_silence_ms=tail_silence_ms,
                trial=trial,
                trace_transcripts=trace_transcripts,
            )
            for trial in range(1, trial_count + 1)
        ]
    )
    commit_to_pcm = [trial.commit_to_first_pcm_ms for trial in trials]
    true_end_to_pcm = [trial.true_end_to_first_pcm_ms for trial in trials]
    return VoiceBenchmarkReport(
        url=url,
        wav_path=str(wav_path),
        trials=trials,
        commit_to_first_pcm_p50_ms=percentile(commit_to_pcm, 0.50),
        commit_to_first_pcm_p90_ms=percentile(commit_to_pcm, 0.90),
        commit_to_first_pcm_p95_ms=percentile(commit_to_pcm, 0.95),
        true_end_to_first_pcm_p50_ms=percentile(true_end_to_pcm, 0.50),
    )


def require_timestamp(value: float | None, label: str) -> float:
    if value is None:
        raise RuntimeError(f"The voice trial did not observe {label}.")
    return value


def milliseconds_between(start_at: float, end_at: float) -> float:
    if end_at < start_at:
        raise RuntimeError("Benchmark timestamps must increase monotonically.")
    return (end_at - start_at) * 1_000


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(math.ceil(quantile * len(ordered)), 1)
    return ordered[rank - 1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a live Voice Light WebSocket.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tail-silence-ms", type=int, default=1_200)
    parser.add_argument("--trace-transcripts", action="store_true")
    arguments = parser.parse_args()
    if arguments.trials <= 0:
        parser.error("--trials must be positive.")
    if arguments.tail_silence_ms < 750:
        parser.error("--tail-silence-ms must cover the VAD and commitment windows.")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    report = asyncio.run(
        run_benchmark(
            url=arguments.url,
            wav_path=arguments.wav,
            trial_count=arguments.trials,
            tail_silence_ms=arguments.tail_silence_ms,
            trace_transcripts=arguments.trace_transcripts,
        )
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
