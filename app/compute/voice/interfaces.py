from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.compute.voice.conversation import ModelMessage
from app.compute.voice.schemas import (
    CapturedAudioChunk,
    InteractionPrediction,
    SpeechUnderstandingEvent,
    TranscriptRevision,
)
from app.compute.voice.tools import SerializedToolCall, ToolCallFailure, ToolSpecification


class SpeechDetector(Protocol):
    def process_audio(self, pcm_bytes: bytes) -> bool: ...


class TranscriptionSession(Protocol):
    async def add_audio(self, pcm_bytes: bytes) -> str | None: ...

    async def finish(self) -> str: ...

    async def close(self) -> None: ...


class Transcriber(Protocol):
    def start_session(self) -> TranscriptionSession: ...

    def close(self) -> None: ...


class LanguageModel(Protocol):
    def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelEvent]: ...


class TextGenerator(Protocol):
    async def generate_text(self, request: TextGenerationRequest) -> str: ...


@dataclass(frozen=True)
class LanguageModelRequest:
    assistant_generation_id: int
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolSpecification, ...]


@dataclass(frozen=True)
class TextGenerationRequest:
    system_prompt: str
    user_prompt: str
    max_new_tokens: int


class LanguageModelEventType(StrEnum):
    SPOKEN_TEXT_DELTA = "spoken_text_delta"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL = "tool_call"
    TOOL_CALL_FAILURE = "tool_call_failure"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class LanguageModelTextDelta:
    text: str
    cumulative_token_count: int
    invocation_id: int
    type: LanguageModelEventType = LanguageModelEventType.SPOKEN_TEXT_DELTA


@dataclass(frozen=True)
class LanguageModelToolCallStarted:
    invocation_id: int
    call_id: str
    cumulative_token_count: int
    type: LanguageModelEventType = LanguageModelEventType.TOOL_CALL_STARTED


@dataclass(frozen=True)
class LanguageModelToolCall:
    invocation_id: int
    request: SerializedToolCall
    cumulative_token_count: int
    type: LanguageModelEventType = LanguageModelEventType.TOOL_CALL


@dataclass(frozen=True)
class LanguageModelToolCallFailure:
    invocation_id: int
    failure: ToolCallFailure
    cumulative_token_count: int
    type: LanguageModelEventType = LanguageModelEventType.TOOL_CALL_FAILURE


@dataclass(frozen=True)
class LanguageModelCompleted:
    invocation_id: int
    cumulative_token_count: int
    type: LanguageModelEventType = LanguageModelEventType.COMPLETED


@dataclass(frozen=True)
class LanguageModelFailed:
    invocation_id: int
    message: str
    type: LanguageModelEventType = LanguageModelEventType.FAILED


LanguageModelEvent = (
    LanguageModelTextDelta
    | LanguageModelToolCallStarted
    | LanguageModelToolCall
    | LanguageModelToolCallFailure
    | LanguageModelCompleted
    | LanguageModelFailed
)


@dataclass(frozen=True)
class TurnPredictionObservation:
    audio_chunk: CapturedAudioChunk
    transcript_revision: TranscriptRevision | None


class TurnPredictionSource(Protocol):
    async def predict(
        self,
        observation: TurnPredictionObservation,
    ) -> InteractionPrediction | None: ...

    async def close(self) -> None: ...


class TurnPredictionProvider(Protocol):
    def create_session(self) -> TurnPredictionSource: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FinalizedSpeechTurn:
    text: str
    transcript_revision: TranscriptRevision | None


class SpeechUnderstandingSession(Protocol):
    @property
    def stream_epoch(self) -> int: ...

    @property
    def turn_epoch(self) -> int: ...

    async def add_audio(self, chunk: CapturedAudioChunk) -> None: ...

    def events(self) -> AsyncIterator[SpeechUnderstandingEvent]: ...

    def drain_events(self) -> tuple[SpeechUnderstandingEvent, ...]: ...

    async def finalize_turn(self) -> FinalizedSpeechTurn: ...

    async def close(self) -> None: ...


class SpeechUnderstandingProvider(Protocol):
    def create_session(self, stream_epoch: int) -> SpeechUnderstandingSession: ...

    def close(self) -> None: ...


class SpeechUnderstandingProviderFactory(Protocol):
    def create(self) -> SpeechUnderstandingProvider: ...


class IntegratedNemotronSpeechUnderstandingSession(SpeechUnderstandingSession, Protocol):
    """Future same-pass Nemotron ASR and turn-adapter conversation state."""


class IntegratedNemotronSpeechUnderstandingProvider(SpeechUnderstandingProvider, Protocol):
    """Future application owner for the integrated Nemotron worker manager."""


@dataclass(frozen=True)
class SynthesisWord:
    text: str
    text_start: int
    text_end: int


@dataclass(frozen=True)
class SynthesizedAudioChunk:
    pcm_bytes: bytes
    start_sample: int


@dataclass(frozen=True)
class SynthesizedWordBoundary:
    text_offset: int
    start_sample: int


@dataclass(frozen=True)
class KyutaiSynthesisFirstAudioMetrics:
    first_word_to_audio_seconds: float
    tokenization_seconds: float
    language_model_step_seconds: float
    mimi_decode_seconds: float
    model_step_count: int
    first_audio_model_step: int


@dataclass(frozen=True)
class VoxtreamSynthesisFirstAudioMetrics:
    first_word_to_audio_seconds: float
    prompt_preparation_seconds: float
    first_frame_generation_seconds: float


SynthesisFirstAudioMetrics = KyutaiSynthesisFirstAudioMetrics | VoxtreamSynthesisFirstAudioMetrics
SynthesisEvent = SynthesizedAudioChunk | SynthesizedWordBoundary | SynthesisFirstAudioMetrics


class SpeechSynthesisSession(Protocol):
    async def add_word(self, word: SynthesisWord) -> None: ...

    async def finish_input(self) -> None: ...

    def stream_events(self) -> AsyncIterator[SynthesisEvent]: ...

    async def cancel(self) -> None: ...


class SpeechSynthesizer(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def start_session(self) -> SpeechSynthesisSession: ...

    def close(self) -> None: ...
