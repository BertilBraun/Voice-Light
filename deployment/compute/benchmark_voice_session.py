from __future__ import annotations

import argparse
import asyncio
import math
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import Field, TypeAdapter
from websockets.asyncio.client import ClientConnection, connect

from app.compute.voice.schemas import (
    AssistantAudioBoundaryEvent,
    AssistantTextDeltaEvent,
    ErrorEvent,
    LlmHistoryEvent,
    PlaybackClockEvent,
    PlaybackCompleteEvent,
    PlaybackStartedEvent,
    PlaybackState,
    SessionReadyEvent,
    SessionStartEvent,
    SessionStopEvent,
    SpeechStateEvent,
    SpeechUnderstandingDebugEvent,
    TranscriptEvent,
    TurnAdapterStatus,
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
    speech_debug_frame_count: int = Field(ge=0)
    turn_prediction_count: int = Field(ge=0)
    adapter_statuses: tuple[TurnAdapterStatus, ...]
    adapter_inference_latency_p50_ms: float | None = Field(default=None, ge=0.0)


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


class PlaybackEventSender(Protocol):
    async def send(self, message: str) -> None: ...


@dataclass
class PlaybackDrainState:
    output_sample_rate: int | None = None
    generation_id: int | None = None
    next_sequence_number: int = 0
    source_sample_position: int = 0
    started: bool = False

    def configure(self, output_sample_rate: int) -> None:
        if output_sample_rate <= 0:
            raise ValueError("Playback output sample rate must be positive.")
        self.output_sample_rate = output_sample_rate

    async def consume_audio(self, sender: PlaybackEventSender, message: bytes) -> int:
        output_sample_rate = self.output_sample_rate
        if output_sample_rate is None:
            raise RuntimeError("Audio arrived before session readiness.")
        generation_id, sequence_number, start_sample = parse_audio_header(message)
        payload = message[12:]
        if not payload or len(payload) % PCM_BYTES_PER_SAMPLE != 0:
            raise ValueError("Voice audio frames must contain complete PCM16 samples.")
        if self.generation_id is None:
            self.generation_id = generation_id
        elif generation_id != self.generation_id:
            raise ValueError("Voice audio changed generation before playback completed.")
        if sequence_number != self.next_sequence_number:
            raise ValueError("Voice audio sequence numbers must be contiguous.")
        if start_sample != self.source_sample_position:
            raise ValueError("Voice audio source sample positions must be contiguous.")
        if not self.started:
            await sender.send(
                PlaybackStartedEvent(
                    generation_id=generation_id,
                    browser_monotonic_time_ns=time.perf_counter_ns(),
                    rendered_output_sample_position=start_sample,
                    source_sample_position=start_sample,
                    output_sample_rate=output_sample_rate,
                ).model_dump_json()
            )
            self.started = True
        sample_count = len(payload) // PCM_BYTES_PER_SAMPLE
        self.next_sequence_number += 1
        self.source_sample_position += sample_count
        await asyncio.sleep(sample_count / output_sample_rate)
        await sender.send(
            PlaybackClockEvent(
                generation_id=generation_id,
                state=PlaybackState.SPEAKING,
                browser_monotonic_time_ns=time.perf_counter_ns(),
                rendered_output_sample_position=self.source_sample_position,
                source_sample_position=self.source_sample_position,
                queued_source_sample_count=0,
                underrun_count=0,
                output_sample_rate=output_sample_rate,
            ).model_dump_json()
        )
        return generation_id

    def complete_event(self, generation_id: int) -> PlaybackCompleteEvent:
        if not self.started or self.generation_id != generation_id:
            raise ValueError("Playback completion must match received audio.")
        output_sample_rate = self.output_sample_rate
        assert output_sample_rate is not None
        return PlaybackCompleteEvent(
            generation_id=generation_id,
            browser_monotonic_time_ns=time.perf_counter_ns(),
            rendered_output_sample_position=self.source_sample_position,
            source_sample_position=self.source_sample_position,
            output_sample_rate=output_sample_rate,
        )


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
    speech_debug_frame_count: int = 0
    turn_prediction_count: int = 0
    adapter_statuses: set[TurnAdapterStatus] = field(default_factory=set)
    adapter_inference_latencies_ms: list[float] = field(default_factory=list)


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
    playback = PlaybackDrainState()
    async with connect(url, max_size=None, open_timeout=180) as websocket:
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
                    generation_id = await playback.consume_audio(websocket, message)
                    if observations.first_pcm_at is None:
                        observations.first_pcm_at = now
                        observations.generation_id = generation_id
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
                            playback.configure(event.output_sample_rate)
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
                        case SpeechUnderstandingDebugEvent():
                            observations.speech_debug_frame_count += 1
                            observations.adapter_statuses.add(event.adapter_status)
                            if event.turn_completion_probability is not None:
                                observations.turn_prediction_count += 1
                                assert event.inference_latency_ms is not None
                                observations.adapter_inference_latencies_ms.append(
                                    event.inference_latency_ms
                                )
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
                                playback.complete_event(event.generation_id).model_dump_json()
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
        speech_debug_frame_count=observations.speech_debug_frame_count,
        turn_prediction_count=observations.turn_prediction_count,
        adapter_statuses=tuple(sorted(observations.adapter_statuses)),
        adapter_inference_latency_p50_ms=(
            None
            if not observations.adapter_inference_latencies_ms
            else percentile(observations.adapter_inference_latencies_ms, 0.50)
        ),
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
