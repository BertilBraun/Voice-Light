from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from app.compute.voice.conversation import (
    ConversationMessage,
    ConversationRole,
    ModelAssistantMessage,
    ModelToolMessage,
    ModelUserMessage,
)
from app.compute.voice.interfaces import (
    LanguageModel,
    LanguageModelCompleted,
    LanguageModelEvent,
    LanguageModelRequest,
    LanguageModelTextDelta,
    LanguageModelToolCall,
    LanguageModelToolCallFailure,
    LanguageModelToolCallStarted,
    SpeechDetector,
    SpeechSynthesisSession,
    SpeechSynthesizer,
    SynthesisEvent,
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
    Transcriber,
    TranscriptionSession,
    TurnPredictionObservation,
    TurnPredictionSource,
)
from app.compute.voice.predictive import (
    CandidateInvalidationReason,
    CandidateOutput,
    PlaybackSink,
    ReleasedAudioChunk,
    ReleasedAudioEnd,
    ReleasedAudioStart,
    ReleasedTextDelta,
    ReleasedWordBoundary,
)
from app.compute.voice.schemas import (
    CausalSource,
    InteractionPrediction,
    PlaybackCommandAcknowledgementEvent,
    PlaybackCommandAction,
    PlaybackCommandEvent,
    PlaybackCompleteEvent,
    PlaybackPauseResult,
    PlaybackProgressEvent,
    PlaybackStartedEvent,
    PlaybackState,
    PlaybackStoppedEvent,
    TraceStamp,
)
from app.compute.voice.search import SearchPipeline, SearchResult
from app.compute.voice.session import SessionPolicy, VoiceSession
from app.compute.voice.speech_understanding import (
    CompositeSpeechUnderstandingProvider,
    SingleSessionTurnPredictionProvider,
)
from app.compute.voice.tools import (
    CurrentLocalTimeHandler,
    PythonArithmeticHandler,
    RuntimeToolRegistry,
    SearchArguments,
    SearchToolHandler,
    SerializedToolCall,
    StandardSearchHandler,
    ToolCallFailure,
    ToolCallFailureReason,
    ToolExecutionFailure,
    ToolExecutionFailureReason,
    ToolExecutor,
    ToolInvalidationReason,
    ToolLifecycle,
    ToolResultCommitStatus,
    ToolSuccess,
)

SPEECH_CHUNK = b"\x01\x00" * 320
SILENCE_CHUNK = b"\x00\x00" * 320
DEFAULT_TEST_POLICY = SessionPolicy(
    silence_duration_ms=40,
    pre_roll_duration_ms=20,
    vad_speculation_enabled=False,
)


@pytest.mark.parametrize(
    ("environment", "expected"),
    (
        ({}, True),
        ({"VOICE_LIGHT_VAD_SPECULATION_ENABLED": "true"}, True),
        ({"VOICE_LIGHT_VAD_SPECULATION_ENABLED": " FALSE "}, False),
    ),
)
def test_session_policy_reads_vad_speculation_switch(
    environment: dict[str, str],
    expected: bool,
) -> None:
    assert SessionPolicy.from_environment(environment).vad_speculation_enabled is expected


def test_session_policy_reads_vad_speculation_debounce() -> None:
    policy = SessionPolicy.from_environment({"VOICE_LIGHT_VAD_SPECULATION_DEBOUNCE_MS": "75"})

    assert policy.vad_speculation_debounce_ms == 75


def test_session_policy_rejects_invalid_vad_speculation_switch() -> None:
    with pytest.raises(
        ValueError,
        match="VOICE_LIGHT_VAD_SPECULATION_ENABLED",
    ):
        SessionPolicy.from_environment({"VOICE_LIGHT_VAD_SPECULATION_ENABLED": "sometimes"})


@pytest.mark.parametrize(
    "value",
    ("soon", "-1"),
)
def test_session_policy_rejects_invalid_vad_speculation_debounce(value: str) -> None:
    with pytest.raises(ValueError, match="VAD speculation debounce|DEBOUNCE_MS"):
        SessionPolicy.from_environment({"VOICE_LIGHT_VAD_SPECULATION_DEBOUNCE_MS": value})


def test_session_policy_rejects_negative_prediction_lag() -> None:
    with pytest.raises(ValueError, match="maximum prediction lag"):
        SessionPolicy(maximum_prediction_lag_ms=-1)


def test_session_policy_reads_tool_timeouts() -> None:
    policy = SessionPolicy.from_environment(
        {
            "VOICE_LIGHT_TOOL_TIMEOUT_SECONDS": "3.5",
            "VOICE_LIGHT_TOOL_CANCELLATION_TIMEOUT_SECONDS": "0.4",
        }
    )

    assert policy.tool_timeout_seconds == 3.5
    assert policy.tool_cancellation_timeout_seconds == 0.4


def test_session_policy_default_allows_search_summarization_latency() -> None:
    assert SessionPolicy.from_environment({}).tool_timeout_seconds == 30.0


def test_session_policy_reads_maximum_tool_rounds() -> None:
    policy = SessionPolicy.from_environment({"VOICE_LIGHT_MAXIMUM_TOOL_ROUNDS": "3"})

    assert policy.maximum_tool_rounds == 3


@pytest.mark.parametrize("value", ("many", "0"))
def test_session_policy_rejects_invalid_maximum_tool_rounds(value: str) -> None:
    with pytest.raises(ValueError, match="maximum tool round|MAXIMUM_TOOL_ROUNDS"):
        SessionPolicy.from_environment({"VOICE_LIGHT_MAXIMUM_TOOL_ROUNDS": value})


@pytest.mark.parametrize(
    "environment_name",
    (
        "VOICE_LIGHT_TOOL_TIMEOUT_SECONDS",
        "VOICE_LIGHT_TOOL_CANCELLATION_TIMEOUT_SECONDS",
    ),
)
def test_session_policy_rejects_nonpositive_tool_timeouts(environment_name: str) -> None:
    with pytest.raises(ValueError, match="tool .*timeout"):
        SessionPolicy.from_environment({environment_name: "0"})


class FakeSpeechDetector:
    def process_audio(self, pcm_bytes: bytes) -> bool:
        return any(pcm_bytes)


class FailingSpeechDetector:
    def process_audio(self, pcm_bytes: bytes) -> bool:
        del pcm_bytes
        raise RuntimeError("synthetic VAD failure")


DEFAULT_TEST_SPEECH_DETECTOR = FakeSpeechDetector()


def audible_conversation(
    request: LanguageModelRequest,
) -> tuple[ConversationMessage, ...]:
    messages: list[ConversationMessage] = []
    for message in request.messages:
        match message:
            case ModelUserMessage():
                messages.append(
                    ConversationMessage(
                        role=ConversationRole.USER,
                        content=message.content,
                    )
                )
            case ModelAssistantMessage() if not message.tool_calls:
                messages.append(
                    ConversationMessage(
                        role=ConversationRole.ASSISTANT,
                        content=message.content,
                    )
                )
            case _:
                raise AssertionError("Expected an audible-only test conversation.")
    return tuple(messages)


class RecordingTranscriber:
    def __init__(self) -> None:
        self.sessions: list[FakeTranscriptionSession] = []

    def start_session(self) -> TranscriptionSession:
        session = FakeTranscriptionSession()
        self.sessions.append(session)
        return session


class ScriptedTranscriber:
    def __init__(
        self,
        partials_by_turn: tuple[tuple[str | None, ...], ...],
        final_texts: tuple[str, ...],
    ) -> None:
        self.partials_by_turn = partials_by_turn
        self.final_texts = final_texts
        self.sessions: list[ScriptedTranscriptionSession] = []

    def start_session(self) -> TranscriptionSession:
        session_index = len(self.sessions)
        partials = (
            self.partials_by_turn[session_index]
            if session_index < len(self.partials_by_turn)
            else ()
        )
        final_text = (
            self.final_texts[session_index] if session_index < len(self.final_texts) else ""
        )
        session = ScriptedTranscriptionSession(partials, final_text)
        self.sessions.append(session)
        return session


class ScriptedTranscriptionSession:
    def __init__(self, partials: tuple[str | None, ...], final_text: str) -> None:
        self.partials = partials
        self.final_text = final_text
        self.next_partial_index = 0
        self.finish_count = 0
        self.closed = False
        self.audio: list[bytes] = []

    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        self.audio.append(pcm_bytes)
        if self.next_partial_index >= len(self.partials):
            return None
        partial = self.partials[self.next_partial_index]
        self.next_partial_index += 1
        return partial

    async def finish(self) -> str:
        self.finish_count += 1
        return self.final_text

    async def close(self) -> None:
        self.closed = True


class FakeTranscriptionSession:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.closed = False

    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        self.audio.append(pcm_bytes)
        return "hello" if any(pcm_bytes) else None

    async def finish(self) -> str:
        return "hello agent"

    async def close(self) -> None:
        self.closed = True


class FakeLanguageModel:
    def __init__(self) -> None:
        self.conversations: list[tuple[ConversationMessage, ...]] = []

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelTextDelta]:
        self.conversations.append(audible_conversation(request))
        yield LanguageModelTextDelta(
            text="One two three four ",
            cumulative_token_count=4,
            invocation_id=1,
        )
        yield LanguageModelTextDelta(
            text="five six seven eight.",
            cumulative_token_count=9,
            invocation_id=1,
        )
        yield LanguageModelCompleted(invocation_id=1, cumulative_token_count=9)


class SplitWordLanguageModel:
    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelTextDelta]:
        del request
        yield LanguageModelTextDelta(text="  Hello", cumulative_token_count=1, invocation_id=1)
        yield LanguageModelTextDelta(text=", wor", cumulative_token_count=3, invocation_id=1)
        yield LanguageModelTextDelta(text="ld! Next", cumulative_token_count=5, invocation_id=1)
        yield LanguageModelCompleted(invocation_id=1, cumulative_token_count=5)


class ScriptedWeatherLanguageModel:
    def __init__(
        self,
        second_pass_answer: str = "It is 12 degrees and lightly cloudy in London.",
    ) -> None:
        self.requests: list[LanguageModelRequest] = []
        self.raw_first_pass = (
            "Let me check that."
            '<tool_call>{"name":"search","arguments":{"query":"current weather in London"}}'
            "</tool_call>"
        )
        self.parsed_first_pass_events: list[LanguageModelEvent] = []
        self.second_pass_answer = second_pass_answer

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelEvent]:
        self.requests.append(request)
        invocation_id = len(self.requests)
        if invocation_id == 1:
            events: tuple[LanguageModelEvent, ...] = (
                LanguageModelTextDelta(
                    invocation_id=1,
                    text="Let me check that.",
                    cumulative_token_count=5,
                ),
                LanguageModelToolCallStarted(
                    invocation_id=1,
                    call_id="qwen-1-tool-1",
                    cumulative_token_count=6,
                ),
                LanguageModelToolCall(
                    invocation_id=1,
                    request=SerializedToolCall(
                        id="qwen-1-tool-1",
                        name="search",
                        arguments_json='{"query":"current weather in London"}',
                    ),
                    cumulative_token_count=16,
                ),
                LanguageModelCompleted(
                    invocation_id=1,
                    cumulative_token_count=17,
                ),
            )
            self.parsed_first_pass_events.extend(events)
            for event in events:
                yield event
            return
        yield LanguageModelTextDelta(
            invocation_id=invocation_id,
            text=self.second_pass_answer,
            cumulative_token_count=12,
        )
        yield LanguageModelCompleted(
            invocation_id=invocation_id,
            cumulative_token_count=12,
        )


class SequentialWeatherLanguageModel:
    def __init__(self) -> None:
        self.requests: list[LanguageModelRequest] = []

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelEvent]:
        self.requests.append(request)
        invocation_id = len(self.requests)
        if request.assistant_generation_id != 1:
            yield LanguageModelTextDelta(
                invocation_id=invocation_id,
                text="Take the heavier coat.",
                cumulative_token_count=5,
            )
            yield LanguageModelCompleted(
                invocation_id=invocation_id,
                cumulative_token_count=5,
            )
            return
        if invocation_id == 1:
            audible_content = "I will check London first."
            call_id = "qwen-1-tool-1"
            location = "London"
        elif invocation_id == 2:
            audible_content = "London is cool; I will compare Berlin."
            call_id = "qwen-2-tool-1"
            location = "Berlin"
        else:
            yield LanguageModelTextDelta(
                invocation_id=invocation_id,
                text="Berlin is warmer at 18 degrees.",
                cumulative_token_count=7,
            )
            yield LanguageModelCompleted(
                invocation_id=invocation_id,
                cumulative_token_count=7,
            )
            return
        yield LanguageModelTextDelta(
            invocation_id=invocation_id,
            text=audible_content,
            cumulative_token_count=6,
        )
        yield LanguageModelToolCallStarted(
            invocation_id=invocation_id,
            call_id=call_id,
            cumulative_token_count=7,
        )
        yield LanguageModelToolCall(
            invocation_id=invocation_id,
            request=SerializedToolCall(
                id=call_id,
                name="search",
                arguments_json=f'{{"query":"current weather in {location}"}}',
            ),
            cumulative_token_count=16,
        )
        yield LanguageModelCompleted(
            invocation_id=invocation_id,
            cumulative_token_count=17,
        )


class InvalidToolCallLanguageModel:
    def __init__(self, failure: ToolCallFailure) -> None:
        self.failure = failure
        self.requests: list[LanguageModelRequest] = []

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelEvent]:
        self.requests.append(request)
        invocation_id = len(self.requests)
        if invocation_id == 1:
            yield LanguageModelTextDelta(
                invocation_id=1,
                text="Let me check that.",
                cumulative_token_count=5,
            )
            yield LanguageModelToolCallFailure(
                invocation_id=1,
                failure=self.failure,
                cumulative_token_count=10,
            )
            yield LanguageModelCompleted(invocation_id=1, cumulative_token_count=10)
            return
        yield LanguageModelTextDelta(
            invocation_id=2,
            text="I could not retrieve the weather right now.",
            cumulative_token_count=9,
        )
        yield LanguageModelCompleted(invocation_id=2, cumulative_token_count=9)


class ControlledWeatherHandler:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.arguments: list[SearchArguments] = []

    async def __call__(self, arguments: SearchArguments) -> str:
        self.arguments.append(arguments)
        self.started.set()
        try:
            await asyncio.to_thread(self.release.wait)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return "London is 12 degrees and lightly cloudy."


class SecondRoundControlledWeatherHandler:
    def __init__(self) -> None:
        self.second_started = threading.Event()
        self.second_cancelled = threading.Event()
        self.arguments: list[SearchArguments] = []
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.release_event: asyncio.Event | None = None

    async def __call__(self, arguments: SearchArguments) -> str:
        self.arguments.append(arguments)
        if arguments.query == "current weather in London":
            return "London is 12 degrees and lightly cloudy."
        self.event_loop = asyncio.get_running_loop()
        self.release_event = asyncio.Event()
        self.second_started.set()
        try:
            await self.release_event.wait()
        except asyncio.CancelledError:
            self.second_cancelled.set()
            raise
        return "Berlin is 18 degrees and clear."

    def release_second(self) -> None:
        if self.event_loop is None or self.release_event is None:
            raise AssertionError("The second weather round has not started.")
        self.event_loop.call_soon_threadsafe(self.release_event.set)


class CancellationResistantWeatherHandler:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    async def __call__(self, arguments: SearchArguments) -> str:
        del arguments
        self.started.set()
        try:
            await asyncio.to_thread(self.release.wait)
        except asyncio.CancelledError:
            self.cancelled.set()
            await asyncio.to_thread(self.release.wait)
        self.finished.set()
        return "London is 12 degrees and lightly cloudy."


class UnconfiguredTestSearchHandler:
    async def __call__(self, arguments: SearchArguments) -> str:
        del arguments
        raise RuntimeError("Search was not configured for this test.")


def create_search_registry(search_handler: SearchToolHandler) -> RuntimeToolRegistry:
    return RuntimeToolRegistry(
        search_handler=search_handler,
        calculate_handler=PythonArithmeticHandler(),
        get_time_handler=CurrentLocalTimeHandler(
            lambda: datetime(2026, 7, 18, tzinfo=timezone.utc)
        ),
    )


class GenerationAwareToolLanguageModel:
    def __init__(self) -> None:
        self.requests: list[LanguageModelRequest] = []
        self.old_continuation_started = threading.Event()

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelEvent]:
        self.requests.append(request)
        invocation_id = len(self.requests)
        if request.assistant_generation_id == 1 and len(request.messages) == 1:
            yield LanguageModelTextDelta(
                invocation_id=invocation_id,
                text="Let me check that.",
                cumulative_token_count=5,
            )
            yield LanguageModelToolCallStarted(
                invocation_id=invocation_id,
                call_id=f"qwen-{invocation_id}-tool-1",
                cumulative_token_count=6,
            )
            yield LanguageModelToolCall(
                invocation_id=invocation_id,
                request=SerializedToolCall(
                    id=f"qwen-{invocation_id}-tool-1",
                    name="search",
                    arguments_json='{"query":"current weather in London"}',
                ),
                cumulative_token_count=16,
            )
            yield LanguageModelCompleted(
                invocation_id=invocation_id,
                cumulative_token_count=17,
            )
            return
        if request.assistant_generation_id == 1:
            self.old_continuation_started.set()
            yield LanguageModelTextDelta(
                invocation_id=invocation_id,
                text="Stale London answer.",
                cumulative_token_count=4,
            )
            yield LanguageModelCompleted(
                invocation_id=invocation_id,
                cumulative_token_count=4,
            )
            return
        yield LanguageModelTextDelta(
            invocation_id=invocation_id,
            text="Replacement answer.",
            cumulative_token_count=3,
        )
        yield LanguageModelCompleted(
            invocation_id=invocation_id,
            cumulative_token_count=3,
        )


class PartialToolCallLanguageModel:
    def __init__(self) -> None:
        self.requests: list[LanguageModelRequest] = []
        self.partial_call_started = threading.Event()
        self.partial_call_cancelled = threading.Event()
        self.release = threading.Event()

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelEvent]:
        self.requests.append(request)
        invocation_id = len(self.requests)
        if request.assistant_generation_id == 1:
            try:
                yield LanguageModelTextDelta(
                    invocation_id=invocation_id,
                    text="Let me check that.",
                    cumulative_token_count=5,
                )
                yield LanguageModelToolCallStarted(
                    invocation_id=invocation_id,
                    call_id=f"qwen-{invocation_id}-tool-1",
                    cumulative_token_count=6,
                )
                self.partial_call_started.set()
                await asyncio.to_thread(self.release.wait)
                yield LanguageModelCompleted(
                    invocation_id=invocation_id,
                    cumulative_token_count=6,
                )
            finally:
                self.partial_call_cancelled.set()
            return
        yield LanguageModelTextDelta(
            invocation_id=invocation_id,
            text="Replacement answer.",
            cumulative_token_count=3,
        )
        yield LanguageModelCompleted(
            invocation_id=invocation_id,
            cumulative_token_count=3,
        )


class SlowLanguageModel:
    def __init__(self) -> None:
        self.conversations: list[tuple[ConversationMessage, ...]] = []

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelTextDelta]:
        self.conversations.append(audible_conversation(request))
        yield LanguageModelTextDelta(
            text="One two three ",
            cumulative_token_count=3,
            invocation_id=1,
        )
        await asyncio.sleep(10)


class CancellationTrackingLanguageModel:
    def __init__(self) -> None:
        self.active_generation_count = 0
        self.overlapped = False

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelTextDelta]:
        del request
        self.active_generation_count += 1
        if self.active_generation_count > 1:
            self.overlapped = True
        try:
            yield LanguageModelTextDelta(
                text="One two three ",
                cumulative_token_count=3,
                invocation_id=1,
            )
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.05)
            self.active_generation_count -= 1


class PredictiveTrackingLanguageModel:
    def __init__(
        self,
        block: bool = False,
        initial_delay_seconds: float = 0.0,
    ) -> None:
        self.block = block
        self.initial_delay_seconds = initial_delay_seconds
        self.conversations: list[tuple[ConversationMessage, ...]] = []
        self.active_generation_count = 0
        self.overlapped = False
        self.cancelled_count = 0
        self.completed_count = 0
        self.generation_started = threading.Event()
        self.first_delta_produced = threading.Event()

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelTextDelta]:
        self.conversations.append(audible_conversation(request))
        self.active_generation_count += 1
        self.overlapped = self.overlapped or self.active_generation_count > 1
        self.generation_started.set()
        completed = False
        try:
            await asyncio.sleep(self.initial_delay_seconds)
            yield LanguageModelTextDelta(
                text="Prepared answer ",
                cumulative_token_count=2,
                invocation_id=1,
            )
            self.first_delta_produced.set()
            if self.block:
                await asyncio.sleep(10)
            yield LanguageModelTextDelta(
                text="complete.",
                cumulative_token_count=4,
                invocation_id=1,
            )
            yield LanguageModelCompleted(invocation_id=1, cumulative_token_count=4)
            completed = True
            self.completed_count += 1
        finally:
            if not completed:
                self.cancelled_count += 1
            try:
                await asyncio.sleep(0.02)
            finally:
                self.active_generation_count -= 1


@dataclass(frozen=True)
class PredictionDirective:
    p_user_speech: float
    p_user_yield: float
    p_user_interruption: float = 0.0
    confidence: float = 1.0


class DeterministicTurnPredictionSource:
    def __init__(
        self,
        directives: tuple[PredictionDirective | None, ...],
    ) -> None:
        self.directives = directives
        self.next_directive_index = 0
        self.observations: list[TurnPredictionObservation] = []
        self.closed = False

    async def predict(
        self,
        observation: TurnPredictionObservation,
    ) -> InteractionPrediction | None:
        self.observations.append(observation)
        if self.next_directive_index >= len(self.directives):
            return None
        directive = self.directives[self.next_directive_index]
        self.next_directive_index += 1
        if directive is None:
            return None
        return create_test_prediction(observation, directive)

    async def close(self) -> None:
        self.closed = True


class DelayedTurnPredictionSource:
    def __init__(self, directive: PredictionDirective) -> None:
        self.directive = directive
        self.started = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()
        self.observation_count = 0
        self.closed = False

    async def predict(
        self,
        observation: TurnPredictionObservation,
    ) -> InteractionPrediction | None:
        self.observation_count += 1
        if self.observation_count > 1:
            return None
        self.started.set()
        await asyncio.to_thread(self.release.wait)
        prediction = create_test_prediction(observation, self.directive)
        self.completed.set()
        return prediction

    async def close(self) -> None:
        self.closed = True
        self.release.set()


class FailingTurnPredictionSource:
    def __init__(self) -> None:
        self.failure_count = 0

    async def predict(
        self,
        observation: TurnPredictionObservation,
    ) -> InteractionPrediction | None:
        del observation
        self.failure_count += 1
        raise RuntimeError("synthetic optional interaction-policy failure")

    async def close(self) -> None:
        return


def create_test_prediction(
    observation: TurnPredictionObservation,
    directive: PredictionDirective,
) -> InteractionPrediction:
    revision = observation.transcript_revision
    chunk = observation.audio_chunk
    return InteractionPrediction(
        stamp=TraceStamp(
            event_id=str(uuid4()),
            parent_event_ids=(() if revision is None else (revision.stamp.event_id,)),
            stream_epoch=chunk.stream_epoch,
            turn_epoch=chunk.turn_epoch,
            inference_step=chunk.sequence_number,
            observation_id=f"audio:{chunk.stream_epoch}:{chunk.sequence_number}",
            observation_monotonic_time_ns=chunk.monotonic_observation_time_ns,
            emission_monotonic_time_ns=chunk.monotonic_observation_time_ns,
            encoder_frame_start=None,
            encoder_frame_end=None,
            input_start_sample=chunk.start_input_sample,
            input_end_sample=chunk.end_input_sample,
            observed_through_input_sample=chunk.end_input_sample,
            input_sample_position=chunk.end_input_sample,
            output_sample_position=None,
            conditioned_transcript_revision_id=(None if revision is None else revision.revision_id),
            conditioned_playback_event_id=chunk.playback_condition.event_id,
            source=CausalSource.TURN_ADAPTER,
            model_name="deterministic-test-source",
            model_revision="1",
        ),
        p_user_speech=directive.p_user_speech,
        p_user_yield=directive.p_user_yield,
        p_user_backchannel=0.0,
        p_user_interruption=directive.p_user_interruption,
        future_user_activity_horizons=(),
        assistant_playback_state=chunk.playback_condition.state,
        confidence=directive.confidence,
    )


class InMemoryPlaybackSink:
    def __init__(self) -> None:
        self.outputs: list[CandidateOutput] = []

    async def send(self, output: CandidateOutput) -> None:
        self.outputs.append(output)


def released_text(sink: InMemoryPlaybackSink) -> str:
    return "".join(output.text for output in sink.outputs if isinstance(output, ReleasedTextDelta))


class FailingLanguageModel:
    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelTextDelta]:
        del request
        raise RuntimeError("synthetic language failure")
        yield


class CancellationFailingLanguageModel:
    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelTextDelta]:
        del request
        try:
            yield LanguageModelTextDelta(
                text="One two three ",
                cumulative_token_count=3,
                invocation_id=1,
            )
            await asyncio.sleep(10)
        finally:
            raise RuntimeError("synthetic language cancellation failure")


class FakeSpeechSynthesisSession:
    def __init__(self, words: list[SynthesisWord]) -> None:
        self.words = words
        self.events: asyncio.Queue[SynthesisEvent | None] = asyncio.Queue()
        self.next_sample = 0
        self.finished = False

    async def add_word(self, word: SynthesisWord) -> None:
        self.words.append(word)
        await self.events.put(
            SynthesizedWordBoundary(text_offset=word.text_end, start_sample=self.next_sample)
        )
        await self.events.put(
            SynthesizedAudioChunk(
                pcm_bytes=b"\x01\x00\x02\x00",
                start_sample=self.next_sample,
            )
        )
        self.next_sample += 2

    async def finish_input(self) -> None:
        self.finished = True
        await self.events.put(None)

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        while (event := await self.events.get()) is not None:
            yield event

    async def cancel(self) -> None:
        if not self.finished:
            self.finished = True
            await self.events.put(None)


class RecordingSpeechSynthesizer:
    def __init__(self) -> None:
        self.sessions: list[FakeSpeechSynthesisSession] = []
        self.words: list[SynthesisWord] = []

    @property
    def sample_rate(self) -> int:
        return 24_000

    def start_session(self) -> SpeechSynthesisSession:
        session = FakeSpeechSynthesisSession(self.words)
        self.sessions.append(session)
        return session


class FailingSpeechSynthesisSession:
    def __init__(self) -> None:
        self.failure_ready = asyncio.Event()

    async def add_word(self, word: SynthesisWord) -> None:
        del word
        self.failure_ready.set()

    async def finish_input(self) -> None:
        self.failure_ready.set()

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        await self.failure_ready.wait()
        raise RuntimeError("synthetic speech failure")
        yield

    async def cancel(self) -> None:
        self.failure_ready.set()


class FailingSpeechSynthesizer:
    @property
    def sample_rate(self) -> int:
        return 24_000

    def start_session(self) -> SpeechSynthesisSession:
        return FailingSpeechSynthesisSession()


class CleanupFailingSpeechSynthesisSession:
    async def add_word(self, word: SynthesisWord) -> None:
        del word

    async def finish_input(self) -> None:
        return

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        await asyncio.sleep(10)
        yield

    async def cancel(self) -> None:
        raise RuntimeError("synthetic cleanup failure")


class CleanupFailingSpeechSynthesizer:
    @property
    def sample_rate(self) -> int:
        return 24_000

    def start_session(self) -> SpeechSynthesisSession:
        return CleanupFailingSpeechSynthesisSession()


def test_full_session_streams_audio_and_commits_naturally_completed_history() -> None:
    language_model = FakeLanguageModel()
    transcriber = RecordingTranscriber()
    synthesizer = RecordingSpeechSynthesizer()
    web_app = create_test_app(transcriber, language_model, synthesizer)

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        ready = websocket.receive_json()
        assert ready["type"] == "session.ready"
        assert ready["output_sample_rate"] == 24_000

        send_turn(websocket)
        first_events, audio_frame = receive_until(websocket, "assistant.audio.end")
        send_playback_started(websocket, 1)
        send_playback_complete(websocket, 1)
        send_turn(websocket)
        second_events, _ = receive_until(websocket, "assistant.audio.end")
        send_playback_complete(websocket, 2)
        websocket.send_json({"type": "session.stop"})

    assert "assistant.audio.text_boundary" in [event["type"] for event in first_events]
    assert second_events[-1]["generation_id"] == 2
    assert audio_frame is not None
    assert struct.unpack("<III", audio_frame[:12]) == (1, 7, 14)
    assert audio_frame[12:] == b"\x01\x00\x02\x00"
    assert language_model.conversations == [
        (ConversationMessage(role=ConversationRole.USER, content="hello agent"),),
        (
            ConversationMessage(role=ConversationRole.USER, content="hello agent"),
            ConversationMessage(
                role=ConversationRole.ASSISTANT,
                content="One two three four five six seven eight.",
            ),
            ConversationMessage(role=ConversationRole.USER, content="hello agent"),
        ),
    ]
    assert len(synthesizer.sessions) == 2
    assert synthesizer.sessions[0] is not synthesizer.sessions[1]


def test_weather_tool_streams_bridge_and_final_answer_in_one_playback_turn() -> None:
    language_model = ScriptedWeatherLanguageModel()
    weather_handler = ControlledWeatherHandler()
    transcriber = RecordingTranscriber()
    synthesizer = RecordingSpeechSynthesizer()
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        synthesizer,
        playback_sink=sink,
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        history_events, _ = receive_until(websocket, "llm.history")
        first_request_events, _ = receive_until(websocket, "llm.model_request")
        assert weather_handler.started.wait(timeout=1)
        wait_until(lambda: any(isinstance(output, ReleasedAudioChunk) for output in sink.outputs))

        first_request_event = first_request_events[-1]
        assert first_request_event["generation_id"] == 1
        assert first_request_event["invocation_index"] == 1
        assert not first_request_event["speculative"]
        assert first_request_event["messages"][0]["role"] == "system"
        assert first_request_event["messages"][1] == {
            "role": "user",
            "content": "hello agent",
        }
        assert first_request_event["tools"][0]["function"]["name"] == "search"
        assert len(language_model.requests) == 1
        assert len(synthesizer.sessions) == 1
        assert synthesizer.sessions[0].finished
        assert weather_handler.arguments == [SearchArguments(query="current weather in London")]
        assert released_text(sink) == "Let me check that."
        assert all("<tool_call>" not in word.text for word in synthesizer.words)
        staged_generation = sessions[0].generations[1]
        staged_tool = staged_generation.tool_executions[0]
        assert staged_tool.identity.assistant_generation_id == 1
        assert staged_tool.identity.user_turn_id == 1
        assert staged_tool.identity.transcript_revision_id is not None
        assert staged_tool.round_index == 1
        assert staged_tool.invocation_id == 1
        assert staged_tool.call_id == "qwen-1-tool-1"
        assert staged_tool.result_commit_status is ToolResultCommitStatus.STAGED
        assert staged_generation.model_context_turn.staged_tool_call is not None
        assert staged_generation.model_context_turn.committed_tool_exchanges == []
        first_bridge_boundary = next(
            output
            for output in sink.outputs
            if isinstance(output, ReleasedWordBoundary) and output.start_sample == 0
        )
        send_playback_progress(
            websocket,
            generation_id=1,
            text_offset=first_bridge_boundary.text_offset,
            boundary_start_sample=0,
            played_sample_count=0,
        )

        weather_handler.release.set()
        wait_until(
            lambda: (
                len(language_model.requests) == 2
                and any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs)
            )
        )
        second_request_events, _ = receive_until(websocket, "llm.model_request")
        second_request_event = second_request_events[-1]
        assert second_request_event["invocation_index"] == 2
        assert second_request_event["messages"][-2]["role"] == "assistant"
        assert second_request_event["messages"][-2]["tool_calls"][0]["function"]["name"] == "search"
        assert second_request_event["messages"][-1] == {
            "role": "tool",
            "tool_call_id": "qwen-1-tool-1",
            "outcome": {
                "outcome": "success",
                "call_id": "qwen-1-tool-1",
                "tool_name": "search",
                "result": "London is 12 degrees and lightly cloudy.",
            },
        }
        committed_tool = sessions[0].generations[1].tool_executions[0]
        assert committed_tool.result_commit_status is ToolResultCommitStatus.SESSION_COMMITTED
        assert committed_tool.result_committed_at is not None
        assert len(sessions[0].generations[1].model_context_turn.committed_tool_exchanges) == 1
        bridge_boundary = next(
            output
            for output in sink.outputs
            if isinstance(output, ReleasedWordBoundary)
            and output.text_offset == len("Let me check that.")
        )
        next_boundary = next(
            output
            for output in sink.outputs
            if isinstance(output, ReleasedWordBoundary)
            and output.start_sample > bridge_boundary.start_sample
        )
        send_playback_progress(
            websocket,
            generation_id=1,
            text_offset=bridge_boundary.text_offset,
            boundary_start_sample=bridge_boundary.start_sample,
            played_sample_count=next_boundary.start_sample,
        )
        wait_until(
            lambda: (
                sessions[0].conversation[-1]
                == ConversationMessage(
                    role=ConversationRole.ASSISTANT,
                    content="Let me check that.",
                )
            )
        )
        send_playback_complete(websocket, generation_id=1)
        wait_until(
            lambda: (
                sessions[0].conversation[-1]
                == ConversationMessage(
                    role=ConversationRole.ASSISTANT,
                    content=("Let me check that. It is 12 degrees and lightly cloudy in London."),
                )
            )
        )

        second_request = language_model.requests[1]
        assistant_message = second_request.messages[-2]
        tool_message = second_request.messages[-1]
        assert isinstance(assistant_message, ModelAssistantMessage)
        assert assistant_message.content == "Let me check that."
        assert assistant_message.tool_calls[0].function.arguments == SearchArguments(
            query="current weather in London"
        )
        assert isinstance(tool_message, ModelToolMessage)
        assert tool_message.tool_call_id == "qwen-1-tool-1"
        assert isinstance(tool_message.outcome, ToolSuccess)
        assert tool_message.outcome.result == "London is 12 degrees and lightly cloudy."
        assert second_request.tools == create_search_registry(weather_handler).specifications
        assert len(synthesizer.sessions) == 2
        assert all(session.finished for session in synthesizer.sessions)

        released_outputs = tuple(sink.outputs)
        assert sum(isinstance(output, ReleasedAudioStart) for output in released_outputs) == 1
        assert sum(isinstance(output, ReleasedAudioEnd) for output in released_outputs) == 1
        assert {output.generation_id for output in released_outputs} == {1}
        boundaries = [
            output for output in released_outputs if isinstance(output, ReleasedWordBoundary)
        ]
        assert [boundary.text_offset for boundary in boundaries] == sorted(
            boundary.text_offset for boundary in boundaries
        )
        assert [boundary.start_sample for boundary in boundaries] == sorted(
            boundary.start_sample for boundary in boundaries
        )
        audio_chunks = [
            output for output in released_outputs if isinstance(output, ReleasedAudioChunk)
        ]
        assert [chunk.sequence_number for chunk in audio_chunks] == list(range(len(audio_chunks)))
        assert [chunk.start_sample for chunk in audio_chunks] == list(
            range(0, len(audio_chunks) * 2, 2)
        )

        public_text = released_text(sink)
        public_history = json.dumps(history_events)
        durable_text = " ".join(message.content for message in sessions[0].conversation)
        for private_fragment in ("<tool_call>", "</tool_call>", '"location"', "qwen-1-tool-1"):
            assert private_fragment not in public_text
            assert private_fragment not in public_history
            assert private_fragment not in durable_text
            assert all(private_fragment not in word.text for word in synthesizer.words)

        generation = sessions[0].generations[1]
        assert generation.invocation_ids == [1, 2]
        assert committed_tool.continuation_invocation_id == 2
        assert generation.latency.first_bridge_text is not None
        assert generation.latency.first_bridge_pcm is not None
        assert generation.latency.tool_call_started_at is not None
        assert generation.latency.tool_call_completed_at is not None
        assert generation.latency.tool_execution_started_at is not None
        assert generation.latency.tool_execution_completed_at is not None
        assert generation.latency.tool_result_committed_at is not None
        assert generation.latency.second_invocation_started_at is not None
        assert generation.latency.first_final_answer_text is not None
        assert generation.latency.first_final_answer_pcm is not None
        assert (
            generation.latency.tool_call_started_at
            <= generation.latency.tool_call_completed_at
            <= generation.latency.tool_execution_started_at
            <= generation.latency.tool_execution_completed_at
            <= generation.latency.tool_result_committed_at
            <= generation.latency.second_invocation_started_at
            <= generation.latency.first_final_answer_text.monotonic_time_seconds
            <= generation.latency.first_final_answer_pcm.monotonic_time_seconds
        )
        assert (
            generation.latency.first_bridge_pcm.monotonic_time_seconds
            < generation.latency.tool_execution_completed_at
        )
        websocket.send_json({"type": "session.stop"})

    assert language_model.raw_first_pass.startswith("Let me check that.")
    assert language_model.parsed_first_pass_events[0] == LanguageModelTextDelta(
        invocation_id=1,
        text="Let me check that.",
        cumulative_token_count=5,
    )
    assert language_model.second_pass_answer.endswith("in London.")


def test_search_raw_results_and_summary_prompt_never_enter_main_model_history() -> None:
    raw_result_marker = "RAW_SEARCH_RESULT_PRIVATE"
    summary_prompt_marker = "SUMMARY_EXCHANGE_PRIVATE"
    final_tool_result = "London is cool and cloudy. Source: https://weather.example"

    class PrivateSearchProvider:
        async def search(
            self,
            query: str,
            result_limit: int,
        ) -> tuple[SearchResult, ...]:
            del query, result_limit
            return (
                SearchResult(
                    title="Weather report",
                    url="https://weather.example",
                    snippet=raw_result_marker,
                ),
            )

        async def close(self) -> None:
            return

    class PrivateSearchSummarizer:
        async def summarize(
            self,
            query: str,
            results: tuple[SearchResult, ...],
        ) -> str:
            del query
            assert results[0].snippet == raw_result_marker
            private_summary_exchange = summary_prompt_marker
            assert private_summary_exchange
            return final_tool_result

    language_model = ScriptedWeatherLanguageModel()
    sessions: list[VoiceSession] = []
    sink = InMemoryPlaybackSink()
    search_handler = StandardSearchHandler(
        SearchPipeline(PrivateSearchProvider(), PrivateSearchSummarizer())
    )
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
        playback_sink=sink,
        created_sessions=sessions,
        tool_executor=create_search_registry(search_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        wait_until(
            lambda: (
                len(language_model.requests) == 2
                and any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs)
            )
        )
        websocket.send_json({"type": "session.stop"})

    continuation_messages = language_model.requests[1].messages
    serialized_messages = repr(continuation_messages)
    assert final_tool_result in serialized_messages
    assert raw_result_marker not in serialized_messages
    assert summary_prompt_marker not in serialized_messages
    assert all(
        raw_result_marker not in message.content and summary_prompt_marker not in message.content
        for message in sessions[0].conversation
    )


def test_sequential_tool_rounds_preserve_context_journal_and_one_playback_turn() -> None:
    language_model = SequentialWeatherLanguageModel()
    weather_handler = SecondRoundControlledWeatherHandler()
    synthesizer = RecordingSpeechSynthesizer()
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    tool_executor = create_search_registry(weather_handler)
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        synthesizer,
        playback_sink=sink,
        created_sessions=sessions,
        tool_executor=tool_executor,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert weather_handler.second_started.wait(timeout=1)
        generation = sessions[0].generations[1]
        wait_until(lambda: len(generation.tool_executions) == 2)

        first_tool, second_tool = generation.tool_executions
        assert first_tool.round_index == 1
        assert first_tool.invocation_id == 1
        assert first_tool.continuation_invocation_id == 2
        assert first_tool.result_commit_status is ToolResultCommitStatus.SESSION_COMMITTED
        assert second_tool.round_index == 2
        assert second_tool.invocation_id == 2
        assert second_tool.result_commit_status is ToolResultCommitStatus.STAGED
        assert len(generation.model_context_turn.committed_tool_exchanges) == 1
        assert generation.model_context_turn.staged_tool_call is not None
        assert released_text(sink) == (
            "I will check London first. London is cool; I will compare Berlin."
        )

        weather_handler.release_second()
        wait_until(lambda: len(language_model.requests) == 3)
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        assert first_tool.continuation_invocation_id == 2
        assert second_tool.continuation_invocation_id == 3
        assert second_tool.result_commit_status is ToolResultCommitStatus.SESSION_COMMITTED
        assert generation.invocation_ids == [1, 2, 3]
        assert len(generation.model_context_turn.committed_tool_exchanges) == 2
        first_exchange, second_exchange = generation.model_context_turn.committed_tool_exchanges
        assert (first_exchange.audible_text_start, first_exchange.audible_text_end) == (
            0,
            len("I will check London first."),
        )
        assert second_exchange.audible_text_start == first_exchange.audible_text_end + 1
        assert second_exchange.audible_text_end == len(
            "I will check London first. London is cool; I will compare Berlin."
        )

        first_continuation_request = language_model.requests[1]
        assert first_continuation_request.tools == tool_executor.specifications
        assert first_continuation_request.messages == (
            ModelUserMessage(content="hello agent"),
            generation.model_context_turn.committed_tool_exchanges[0].assistant_message,
            generation.model_context_turn.committed_tool_exchanges[0].tool_message,
        )
        second_continuation_request = language_model.requests[2]
        assert second_continuation_request.tools == tool_executor.specifications
        assert second_continuation_request.messages == (
            ModelUserMessage(content="hello agent"),
            generation.model_context_turn.committed_tool_exchanges[0].assistant_message,
            generation.model_context_turn.committed_tool_exchanges[0].tool_message,
            generation.model_context_turn.committed_tool_exchanges[1].assistant_message,
            generation.model_context_turn.committed_tool_exchanges[1].tool_message,
        )

        send_playback_complete(websocket, generation_id=1)
        wait_until(lambda: sessions[0].active_generation is None)
        assert sessions[0].conversation[-1] == ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=(
                "I will check London first. London is cool; I will compare Berlin. "
                "Berlin is warmer at 18 degrees."
            ),
        )
        first_generation_outputs = tuple(
            output for output in sink.outputs if output.generation_id == 1
        )
        assert (
            sum(isinstance(output, ReleasedAudioStart) for output in first_generation_outputs) == 1
        )
        assert sum(isinstance(output, ReleasedAudioEnd) for output in first_generation_outputs) == 1
        boundaries = [
            output
            for output in first_generation_outputs
            if isinstance(output, ReleasedWordBoundary)
        ]
        assert [boundary.text_offset for boundary in boundaries] == sorted(
            boundary.text_offset for boundary in boundaries
        )
        assert [boundary.start_sample for boundary in boundaries] == sorted(
            boundary.start_sample for boundary in boundaries
        )
        assert len(synthesizer.sessions) == 3
        assert all(session.finished for session in synthesizer.sessions)

        send_turn(websocket)
        receive_until(websocket, "llm.history")
        wait_until(lambda: len(language_model.requests) == 4)
        later_request = language_model.requests[3]
        wait_until(
            lambda: any(
                isinstance(output, ReleasedAudioEnd) and output.generation_id == 2
                for output in sink.outputs
            )
        )
        send_playback_complete(websocket, generation_id=2)
        wait_until(lambda: sessions[0].active_generation is None)
        websocket.send_json({"type": "session.stop"})

    first_exchange, second_exchange = generation.model_context_turn.committed_tool_exchanges
    assert later_request.messages == (
        ModelUserMessage(content="hello agent"),
        first_exchange.assistant_message,
        first_exchange.tool_message,
        second_exchange.assistant_message,
        second_exchange.tool_message,
        ModelAssistantMessage(content="Berlin is warmer at 18 degrees."),
        ModelUserMessage(content="hello agent"),
    )
    assert later_request.tools == tool_executor.specifications
    assert weather_handler.arguments == [
        SearchArguments(query="current weather in London"),
        SearchArguments(query="current weather in Berlin"),
    ]
    public_text = released_text(sink)
    durable_text = " ".join(message.content for message in sessions[0].conversation)
    for private_fragment in ("<tool_call>", "</tool_call>", '"query"', "qwen-1-tool-1"):
        assert private_fragment not in public_text
        assert private_fragment not in durable_text
        assert all(private_fragment not in word.text for word in synthesizer.words)


def test_tool_round_limit_disables_tools_without_dropping_committed_exchanges() -> None:
    language_model = SequentialWeatherLanguageModel()
    weather_handler = SecondRoundControlledWeatherHandler()
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
        policy=SessionPolicy(
            silence_duration_ms=40,
            pre_roll_duration_ms=20,
            vad_speculation_enabled=False,
            maximum_tool_rounds=2,
        ),
        playback_sink=sink,
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert weather_handler.second_started.wait(timeout=1)
        weather_handler.release_second()
        wait_until(lambda: len(language_model.requests) == 3)
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        generation = sessions[0].generations[1]

        assert language_model.requests[2].tools == ()
        assert len(generation.tool_executions) == 2
        assert len(generation.model_context_turn.committed_tool_exchanges) == 2
        send_playback_complete(websocket, generation_id=1)
        wait_until(lambda: sessions[0].active_generation is None)
        websocket.send_json({"type": "session.stop"})


def test_later_user_turn_receives_prior_structured_tool_exchange() -> None:
    language_model = ScriptedWeatherLanguageModel()
    weather_handler = ControlledWeatherHandler()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert weather_handler.started.wait(timeout=1)
        weather_handler.release.set()
        receive_until(websocket, "assistant.audio.end")
        send_playback_complete(websocket, generation_id=1)
        wait_until(lambda: sessions[0].active_generation is None)

        send_turn(websocket)
        receive_until(websocket, "llm.history")
        wait_until(lambda: len(language_model.requests) == 3)
        later_request = language_model.requests[2]
        websocket.send_json({"type": "session.stop"})

    assert later_request.assistant_generation_id == 2
    assert later_request.messages[0] == ModelUserMessage(content="hello agent")
    assistant_call = later_request.messages[1]
    assert isinstance(assistant_call, ModelAssistantMessage)
    assert assistant_call.content == "Let me check that."
    assert assistant_call.tool_calls[0].id == "qwen-1-tool-1"
    tool_result = later_request.messages[2]
    assert isinstance(tool_result, ModelToolMessage)
    assert tool_result.tool_call_id == assistant_call.tool_calls[0].id
    assert isinstance(tool_result.outcome, ToolSuccess)
    assert later_request.messages[3] == ModelAssistantMessage(
        content="It is 12 degrees and lightly cloudy in London."
    )
    assert later_request.messages[4] == ModelUserMessage(content="hello agent")
    assert later_request.tools == create_search_registry(weather_handler).specifications


def test_invalid_tool_call_is_private_and_produces_spoken_recovery() -> None:
    failure = ToolCallFailure(
        call_id="qwen-1-tool-1",
        reason=ToolCallFailureReason.MALFORMED_JSON,
        message="The tool call was malformed.",
        attempted_tool_name=None,
    )
    language_model = InvalidToolCallLanguageModel(failure)
    weather_handler = ControlledWeatherHandler()
    sink = InMemoryPlaybackSink()
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
        playback_sink=sink,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        websocket.send_json({"type": "session.stop"})

    assert not weather_handler.started.is_set()
    assert released_text(sink) == "Let me check that. I could not retrieve the weather right now."
    tool_message = language_model.requests[1].messages[-1]
    assert isinstance(tool_message, ModelToolMessage)
    assert isinstance(tool_message.outcome, ToolExecutionFailure)
    assert tool_message.outcome.reason is ToolCallFailureReason.MALFORMED_JSON
    assert "<tool_call>" not in released_text(sink)


@pytest.mark.parametrize(
    "failure_mode",
    ("handler", "timeout"),
)
def test_tool_execution_failure_is_typed_and_does_not_fabricate_weather(
    failure_mode: str,
) -> None:
    language_model = ScriptedWeatherLanguageModel(
        second_pass_answer="I could not retrieve the weather right now."
    )
    controlled_handler = ControlledWeatherHandler()

    async def failing_handler(arguments: SearchArguments) -> str:
        del arguments
        raise RuntimeError("synthetic weather failure")

    tool_executor = create_search_registry(
        failing_handler if failure_mode == "handler" else controlled_handler
    )
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    policy = SessionPolicy(
        silence_duration_ms=40,
        pre_roll_duration_ms=20,
        vad_speculation_enabled=False,
        tool_timeout_seconds=1.0 if failure_mode == "handler" else 0.01,
    )
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
        policy=policy,
        playback_sink=sink,
        created_sessions=sessions,
        tool_executor=tool_executor,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        controlled_handler.release.set()
        websocket.send_json({"type": "session.stop"})

    tool_message = language_model.requests[1].messages[-1]
    assert isinstance(tool_message, ModelToolMessage)
    assert isinstance(tool_message.outcome, ToolExecutionFailure)
    expected_reason = (
        ToolExecutionFailureReason.HANDLER_FAILURE
        if failure_mode == "handler"
        else ToolExecutionFailureReason.TIMEOUT
    )
    assert tool_message.outcome.reason is expected_reason
    journal_entry = sessions[0].tool_execution_journal[0]
    assert journal_entry.result_commit_status is ToolResultCommitStatus.SESSION_COMMITTED
    assert journal_entry.outcome == tool_message.outcome
    assert released_text(sink).endswith("I could not retrieve the weather right now.")
    assert "12 degrees" not in released_text(sink)


def test_user_cancellation_while_tool_runs_discards_late_result() -> None:
    language_model = GenerationAwareToolLanguageModel()
    weather_handler = CancellationResistantWeatherHandler()
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
        playback_sink=sink,
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert weather_handler.started.wait(timeout=1)
        wait_until(lambda: released_text(sink) == "Let me check that.")

        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert weather_handler.cancelled.wait(timeout=1)
        wait_until(
            lambda: any(request.assistant_generation_id == 2 for request in language_model.requests)
        )
        weather_handler.release.set()
        assert weather_handler.finished.wait(timeout=1)
        wait_until(
            lambda: any(
                isinstance(output, ReleasedAudioEnd) and output.generation_id == 2
                for output in sink.outputs
            )
        )

        assert not language_model.old_continuation_started.is_set()
        generation_one_text = "".join(
            output.text
            for output in sink.outputs
            if isinstance(output, ReleasedTextDelta) and output.generation_id == 1
        )
        assert generation_one_text == "Let me check that."
        assert "Stale London answer" not in released_text(sink)
        assert all(
            not (
                isinstance(message, ConversationMessage)
                and "Stale London answer" in message.content
            )
            for message in sessions[0].conversation
        )
        first_tool = sessions[0].generations[1].tool_executions[0]
        assert first_tool.lifecycle is ToolLifecycle.CANCELLED
        assert first_tool.result_commit_status is ToolResultCommitStatus.DISCARDED
        assert first_tool.invalidation_reason is ToolInvalidationReason.USER_ACTIVITY
        assert isinstance(first_tool.outcome, ToolExecutionFailure)
        assert first_tool.outcome.reason is ToolExecutionFailureReason.CANCELLED
        assert sessions[0].generations[1].model_context_turn.staged_tool_call is None
        assert sessions[0].generations[1].model_context_turn.committed_tool_exchanges == []
        assert all(
            not (isinstance(message, ModelToolMessage) and message.tool_call_id == "qwen-1-tool-1")
            for message in sessions[0].model_context.snapshot()
        )
        assert first_tool.wasted
        websocket.send_json({"type": "session.stop"})


def test_cancellation_during_later_tool_round_keeps_earlier_committed_exchange() -> None:
    language_model = SequentialWeatherLanguageModel()
    weather_handler = SecondRoundControlledWeatherHandler()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert weather_handler.second_started.wait(timeout=1)
        generation = sessions[0].generations[1]
        wait_until(lambda: len(generation.tool_executions) == 2)

        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert weather_handler.second_cancelled.wait(timeout=1)
        weather_handler.release_second()
        wait_until(
            lambda: any(request.assistant_generation_id == 2 for request in language_model.requests)
        )

        first_tool, second_tool = generation.tool_executions
        assert first_tool.result_commit_status is ToolResultCommitStatus.SESSION_COMMITTED
        assert first_tool.lifecycle is ToolLifecycle.SUCCEEDED
        assert second_tool.result_commit_status is ToolResultCommitStatus.DISCARDED
        assert second_tool.lifecycle is ToolLifecycle.CANCELLED
        assert second_tool.invalidation_reason is ToolInvalidationReason.USER_ACTIVITY
        assert isinstance(second_tool.outcome, ToolExecutionFailure)
        assert second_tool.outcome.reason is ToolExecutionFailureReason.CANCELLED
        assert generation.model_context_turn.staged_tool_call is None
        assert len(generation.model_context_turn.committed_tool_exchanges) == 1
        first_exchange = generation.model_context_turn.committed_tool_exchanges[0]
        private_messages = sessions[0].model_context.snapshot()
        assert first_exchange.assistant_message in private_messages
        assert first_exchange.tool_message in private_messages
        assert all(
            not (isinstance(message, ModelToolMessage) and message.tool_call_id == "qwen-2-tool-1")
            for message in private_messages
        )
        websocket.send_json({"type": "session.stop"})


def test_user_cancellation_while_call_is_buffering_never_starts_tool() -> None:
    language_model = PartialToolCallLanguageModel()
    weather_handler = ControlledWeatherHandler()
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
        playback_sink=sink,
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert language_model.partial_call_started.wait(timeout=1)
        wait_until(lambda: released_text(sink) == "Let me check that.")

        send_turn(websocket)
        receive_until(websocket, "llm.history")
        assert language_model.partial_call_cancelled.wait(timeout=1)
        wait_until(
            lambda: any(request.assistant_generation_id == 2 for request in language_model.requests)
        )
        language_model.release.set()
        assert not weather_handler.started.is_set()
        first_generation_text = "".join(
            output.text
            for output in sink.outputs
            if isinstance(output, ReleasedTextDelta) and output.generation_id == 1
        )
        assert first_generation_text == "Let me check that."
        assert "<tool" not in released_text(sink)
        first_tool = next(
            entry
            for entry in sessions[0].tool_execution_journal
            if entry.identity.assistant_generation_id == 1
        )
        assert first_tool.result_commit_status is ToolResultCommitStatus.NOT_STAGED
        assert first_tool.lifecycle is ToolLifecycle.CANCELLED
        assert isinstance(first_tool.outcome, ToolExecutionFailure)
        assert first_tool.outcome.reason is ToolExecutionFailureReason.CANCELLED
        websocket.send_json({"type": "session.stop"})


def test_words_are_forwarded_on_whitespace_and_trailing_word_is_flushed() -> None:
    synthesizer = RecordingSpeechSynthesizer()
    web_app = create_test_app(
        RecordingTranscriber(),
        SplitWordLanguageModel(),
        synthesizer,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        events, _ = receive_until(websocket, "assistant.audio.end")
        websocket.send_json({"type": "session.stop"})

    assert synthesizer.words == [
        SynthesisWord(text="Hello,", text_start=2, text_end=8),
        SynthesisWord(text="world!", text_start=9, text_end=15),
        SynthesisWord(text="Next", text_start=16, text_end=20),
    ]
    boundaries = [event for event in events if event["type"] == "assistant.audio.text_boundary"]
    assert boundaries == [
        {
            "type": "assistant.audio.text_boundary",
            "generation_id": 1,
            "text_offset": 8,
            "start_sample": 0,
        },
        {
            "type": "assistant.audio.text_boundary",
            "generation_id": 1,
            "text_offset": 15,
            "start_sample": 2,
        },
        {
            "type": "assistant.audio.text_boundary",
            "generation_id": 1,
            "text_offset": 20,
            "start_sample": 4,
        },
    ]


def test_synthesis_failure_cancels_generation_and_reaches_client() -> None:
    web_app = create_test_app(
        RecordingTranscriber(),
        SlowLanguageModel(),
        FailingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        events, _ = receive_until(websocket, "error")
        websocket.send_json({"type": "session.stop"})

    assert "playback.command" in [event["type"] for event in events[:-1]]
    assert events[-1]["type"] == "error"
    assert events[-1]["message"] == ("Response generation failed: synthetic speech failure")
    assert events[-1]["component"] == "speech_synthesis"
    assert events[-1]["operation"] == "stream_synthesis"
    assert events[-1]["generation_id"] == 1
    assert events[-1]["retryable"] is True


def test_successor_generation_waits_for_cancelled_generation_teardown() -> None:
    language_model = CancellationTrackingLanguageModel()
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "playback.command")
        send_playback_stopped(websocket, 1, text_offset=0, played_sample_count=0)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "assistant.text.delta")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["generation_id"] == 2
    assert language_model.overlapped is False


@pytest.mark.parametrize(
    ("partial_text", "final_text"),
    [
        (None, ""),
        ("mm-hm", "mm-hm"),
        ("haha", "haha"),
        ("okay", "okay"),
    ],
)
def test_non_floor_taking_overlap_resumes_existing_generation_without_history(
    partial_text: str | None,
    final_text: str,
) -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", None, None), (partial_text, None)),
        final_texts=("hello agent", final_text),
    )
    language_model = SlowLanguageModel()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        send_playback_started(websocket, 1)
        wait_until(lambda: sessions[0].playback_condition.state is PlaybackState.SPEAKING)

        websocket.send_bytes(SPEECH_CHUNK)
        duck = receive_playback_command(websocket)
        pause = receive_playback_command(websocket)
        assert duck.action is PlaybackCommandAction.DUCK
        assert pause.action is PlaybackCommandAction.PAUSE_AT_BOUNDARY
        send_playback_command_acknowledgement(
            websocket,
            duck,
            resulting_state=PlaybackState.DUCKING,
            pause_result=PlaybackPauseResult.NOT_REQUESTED,
            source_sample_position=1,
        )
        paused_source_position = pause.requested_boundary_source_sample_position or 1
        send_playback_command_acknowledgement(
            websocket,
            pause,
            resulting_state=PlaybackState.PAUSED_BUFFERED,
            pause_result=(
                PlaybackPauseResult.WORD_BOUNDARY
                if pause.requested_boundary_source_sample_position is not None
                else PlaybackPauseResult.FORCED_SAMPLE
            ),
            source_sample_position=paused_source_position,
        )
        websocket.send_bytes(SILENCE_CHUNK)
        resume = receive_playback_command(websocket)
        assert resume.action is PlaybackCommandAction.RESUME
        assert resume.generation_id == 1
        send_playback_command_acknowledgement(
            websocket,
            resume,
            resulting_state=PlaybackState.RESUMING,
            pause_result=PlaybackPauseResult.NOT_REQUESTED,
            source_sample_position=paused_source_position,
        )
        assert len(language_model.conversations) == 1
        assert all(message.content != final_text for message in sessions[0].conversation)
        metrics = sessions[0].overlap_metrics.report()
        assert metrics.cooperative_overlap_count == 1
        assert metrics.user_onset_to_duck_p95_ms is not None
        assert metrics.user_onset_to_pause_p95_ms is not None
        assert metrics.user_onset_to_resume_p95_ms is not None
        websocket.send_json({"type": "session.stop"})

    assert transcriber.sessions[1].finish_count == 1


def test_backchannel_during_tool_wait_preserves_tool_and_resumes_same_generation() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("weather", None, None), ("mm-hm", None)),
        final_texts=("weather in London", "mm-hm"),
    )
    language_model = ScriptedWeatherLanguageModel()
    weather_handler = ControlledWeatherHandler()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        assert weather_handler.started.wait(timeout=1)
        send_playback_started(websocket, 1)
        wait_until(lambda: sessions[0].playback_condition.state is PlaybackState.SPEAKING)

        websocket.send_bytes(SPEECH_CHUNK)
        duck = receive_playback_command(websocket)
        pause = receive_playback_command(websocket)
        send_playback_command_acknowledgement(
            websocket,
            duck,
            resulting_state=PlaybackState.DUCKING,
            pause_result=PlaybackPauseResult.NOT_REQUESTED,
            source_sample_position=1,
        )
        paused_source_position = pause.requested_boundary_source_sample_position or 1
        send_playback_command_acknowledgement(
            websocket,
            pause,
            resulting_state=PlaybackState.PAUSED_BUFFERED,
            pause_result=(
                PlaybackPauseResult.WORD_BOUNDARY
                if pause.requested_boundary_source_sample_position is not None
                else PlaybackPauseResult.FORCED_SAMPLE
            ),
            source_sample_position=paused_source_position,
        )
        websocket.send_bytes(SILENCE_CHUNK)
        resume = receive_playback_command(websocket)
        assert resume.action is PlaybackCommandAction.RESUME
        assert resume.generation_id == 1
        send_playback_command_acknowledgement(
            websocket,
            resume,
            resulting_state=PlaybackState.RESUMING,
            pause_result=PlaybackPauseResult.NOT_REQUESTED,
            source_sample_position=paused_source_position,
        )

        weather_handler.release.set()
        receive_until(websocket, "assistant.audio.end")
        assert len(language_model.requests) == 2
        assert {request.assistant_generation_id for request in language_model.requests} == {1}
        tool = sessions[0].generations[1].tool_executions[0]
        assert tool.lifecycle is ToolLifecycle.SUCCEEDED
        assert all(message.content != "mm-hm" for message in sessions[0].conversation)
        websocket.send_json({"type": "session.stop"})


@pytest.mark.parametrize("interruption_text", ["How?", "No, wait", "Yeah, but this is wrong"])
def test_lexical_interruption_cancels_and_becomes_a_durable_user_turn(
    interruption_text: str,
) -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", None, None), (interruption_text, None, None)),
        final_texts=("hello agent", interruption_text),
    )
    language_model = SlowLanguageModel()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        send_playback_started(websocket, 1)
        wait_until(lambda: sessions[0].playback_condition.state is PlaybackState.SPEAKING)

        websocket.send_bytes(SPEECH_CHUNK)
        assert receive_playback_command(websocket).action is PlaybackCommandAction.DUCK
        assert receive_playback_command(websocket).action is PlaybackCommandAction.PAUSE_AT_BOUNDARY
        cancel = receive_playback_command(websocket)
        assert cancel.action is PlaybackCommandAction.CANCEL
        send_playback_command_acknowledgement(
            websocket,
            cancel,
            resulting_state=PlaybackState.CANCELLED,
            pause_result=PlaybackPauseResult.NOT_REQUESTED,
            source_sample_position=1,
        )
        assert sessions[0].overlap_metrics.report().explicit_stop_p95_ms is not None
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["generation_id"] == 2
    assert events[-1]["messages"][-1] == {
        "role": "user",
        "content": interruption_text,
    }
    assert language_model.conversations[-1][-1] == ConversationMessage(
        role=ConversationRole.USER,
        content=interruption_text,
    )
    assert transcriber.sessions[1].audio[0] == SPEECH_CHUNK


def test_response_requiring_overlap_cancels_running_tool() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("weather", None, None), ("How?", None, None)),
        final_texts=("weather in London", "How?"),
    )
    language_model = GenerationAwareToolLanguageModel()
    weather_handler = CancellationResistantWeatherHandler()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        assert weather_handler.started.wait(timeout=1)
        send_playback_started(websocket, 1)
        wait_until(lambda: sessions[0].playback_condition.state is PlaybackState.SPEAKING)

        websocket.send_bytes(SPEECH_CHUNK)
        assert receive_playback_command(websocket).action is PlaybackCommandAction.DUCK
        assert receive_playback_command(websocket).action is PlaybackCommandAction.PAUSE_AT_BOUNDARY
        cancel = receive_playback_command(websocket)
        assert cancel.action is PlaybackCommandAction.CANCEL
        send_playback_command_acknowledgement(
            websocket,
            cancel,
            resulting_state=PlaybackState.CANCELLED,
            pause_result=PlaybackPauseResult.NOT_REQUESTED,
            source_sample_position=1,
        )
        assert weather_handler.cancelled.wait(timeout=1)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        assert events[-1]["generation_id"] == 2

        weather_handler.release.set()
        assert weather_handler.finished.wait(timeout=1)
        receive_until(websocket, "assistant.audio.end")
        assert not language_model.old_continuation_started.is_set()
        first_tool = sessions[0].generations[1].tool_executions[0]
        assert first_tool.lifecycle is ToolLifecycle.CANCELLED
        assert first_tool.result_commit_status is ToolResultCommitStatus.DISCARDED
        assert first_tool.invalidation_reason is ToolInvalidationReason.RESPONSE_REQUIRING_OVERLAP
        websocket.send_json({"type": "session.stop"})


def test_final_lexical_interruption_never_resumes_provisional_backchannel() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", None, None), (None, None)),
        final_texts=("hello agent", "How?"),
    )
    language_model = SlowLanguageModel()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        send_playback_started(websocket, 1)
        wait_until(lambda: sessions[0].playback_condition.state is PlaybackState.SPEAKING)

        websocket.send_bytes(SPEECH_CHUNK)
        duck = receive_playback_command(websocket)
        pause = receive_playback_command(websocket)
        assert duck.action is PlaybackCommandAction.DUCK
        assert pause.action is PlaybackCommandAction.PAUSE_AT_BOUNDARY
        websocket.send_bytes(SILENCE_CHUNK)
        cancel = receive_playback_command(websocket)
        assert cancel.action is PlaybackCommandAction.CANCEL
        assert all(
            record.command.action is not PlaybackCommandAction.RESUME
            for record in sessions[0].playback_controller.command_records.values()
        )
        send_playback_command_acknowledgement(
            websocket,
            cancel,
            resulting_state=PlaybackState.CANCELLED,
            pause_result=PlaybackPauseResult.NOT_REQUESTED,
            source_sample_position=1,
        )
        events, _ = receive_until(websocket, "llm.history")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["generation_id"] == 2
    assert events[-1]["messages"][-1] == {
        "role": "user",
        "content": "How?",
    }


def test_sustained_transcript_free_overlap_yields_at_500_milliseconds() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", None, None), tuple(None for _ in range(30))),
        final_texts=("hello agent", "continued request"),
    )
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        SlowLanguageModel(),
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        send_playback_started(websocket, 1)
        wait_until(lambda: sessions[0].playback_condition.state is PlaybackState.SPEAKING)

        for _ in range(25):
            websocket.send_bytes(SPEECH_CHUNK)
        assert receive_playback_command(websocket).action is PlaybackCommandAction.DUCK
        assert receive_playback_command(websocket).action is PlaybackCommandAction.PAUSE_AT_BOUNDARY
        cancel = receive_playback_command(websocket)
        assert cancel.action is PlaybackCommandAction.CANCEL
        send_playback_command_acknowledgement(
            websocket,
            cancel,
            resulting_state=PlaybackState.CANCELLED,
            pause_result=PlaybackPauseResult.NOT_REQUESTED,
            source_sample_position=1,
        )
        assert sessions[0].overlap_metrics.report().competitive_overlap_count == 1
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["generation_id"] == 2
    assert events[-1]["messages"][-1]["content"] == "continued request"


def test_overlap_pause_command_marks_missing_word_boundary_for_forced_pause() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", None, None), (None, None)),
        final_texts=("hello agent", ""),
    )
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        SlowLanguageModel(),
        RecordingSpeechSynthesizer(),
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        send_playback_started(websocket, 1)
        wait_until(lambda: sessions[0].playback_condition.state is PlaybackState.SPEAKING)
        generation = sessions[0].active_generation
        assert generation is not None
        generation.boundary_samples.clear()

        websocket.send_bytes(SPEECH_CHUNK)
        assert receive_playback_command(websocket).action is PlaybackCommandAction.DUCK
        pause = receive_playback_command(websocket)
        assert pause.action is PlaybackCommandAction.PAUSE_AT_BOUNDARY
        assert pause.requested_boundary_source_sample_position is None
        assert pause.rendered_output_sample_deadline is not None
        websocket.send_json({"type": "session.stop"})


def test_optional_interaction_failure_keeps_fallback_playback_cancellation() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", None, None), tuple(None for _ in range(30))),
        final_texts=("hello agent", "continued request"),
    )
    prediction_source = FailingTurnPredictionSource()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        SlowLanguageModel(),
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        send_playback_started(websocket, 1)
        wait_until(lambda: sessions[0].playback_condition.state is PlaybackState.SPEAKING)
        for _ in range(25):
            websocket.send_bytes(SPEECH_CHUNK)
        assert receive_playback_command(websocket).action is PlaybackCommandAction.DUCK
        assert receive_playback_command(websocket).action is PlaybackCommandAction.PAUSE_AT_BOUNDARY
        assert receive_playback_command(websocket).action is PlaybackCommandAction.CANCEL
        websocket.send_json({"type": "session.stop"})

    assert prediction_source.failure_count == 1
    assert transcriber.sessions[1].audio


def test_language_model_failure_reaches_client_with_component_context() -> None:
    web_app = create_test_app(
        RecordingTranscriber(),
        FailingLanguageModel(),
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        events, _ = receive_until(websocket, "error")
        websocket.send_json({"type": "session.stop"})

    assert events[-1] == {
        "type": "error",
        "component": "language_model",
        "operation": "generate_text",
        "generation_id": 1,
        "retryable": True,
        "message": "Response generation failed: synthetic language failure",
    }


def test_synthesis_cleanup_failure_does_not_mask_language_model_failure() -> None:
    web_app = create_test_app(
        RecordingTranscriber(),
        FailingLanguageModel(),
        CleanupFailingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        events, _ = receive_until(websocket, "error")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["component"] == "language_model"
    assert events[-1]["message"] == ("Response generation failed: synthetic language failure")


def test_synthesis_cancellation_failure_reaches_client() -> None:
    language_model = CancellationTrackingLanguageModel()
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        CleanupFailingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.text.delta")
        websocket.send_bytes(SPEECH_CHUNK)
        events, _ = receive_until(websocket, "error")
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "assistant.text.delta")
        websocket.send_json({"type": "session.stop"})

    assert "playback.command" in [event["type"] for event in events[:-1]]
    assert events[-1]["type"] == "error"
    assert events[-1]["component"] == "speech_synthesis"
    assert events[-1]["message"] == (
        "Response generation failed: Speech synthesis cleanup failed: synthetic cleanup failure"
    )
    assert language_model.overlapped is False


def test_language_model_cancellation_failure_keeps_component_context() -> None:
    web_app = create_test_app(
        RecordingTranscriber(),
        CancellationFailingLanguageModel(),
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.text.delta")
        websocket.send_bytes(SPEECH_CHUNK)
        events, _ = receive_until(websocket, "error")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["component"] == "language_model"
    assert events[-1]["operation"] == "generate_text"
    assert events[-1]["message"] == (
        "Response generation failed: synthetic language cancellation failure"
    )


def test_speech_detection_failure_reaches_client_with_component_context() -> None:
    web_app = create_test_app(
        RecordingTranscriber(),
        FakeLanguageModel(),
        RecordingSpeechSynthesizer(),
        speech_detector=FailingSpeechDetector(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        events, _ = receive_until(websocket, "error")

    assert events[-1]["component"] == "speech_detection"
    assert events[-1]["operation"] == "detect_speech"
    assert events[-1]["message"] == "synthetic VAD failure"


def test_canceled_generation_accepts_final_browser_acknowledgement() -> None:
    language_model = SlowLanguageModel()
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")

        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "playback.command")
        send_playback_stopped(websocket, 1, text_offset=3, played_sample_count=3)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["messages"] == [
        {"role": "user", "content": "hello agent"},
        {"role": "assistant", "content": "One..."},
        {"role": "user", "content": "hello agent"},
    ]


def test_invalid_or_stale_playback_progress_is_ignored() -> None:
    language_model = SlowLanguageModel()
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        send_playback_progress(
            websocket,
            generation_id=1,
            text_offset=7,
            boundary_start_sample=99,
            played_sample_count=100,
        )
        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "playback.command")
        send_playback_stopped(websocket, 1, text_offset=0, played_sample_count=0)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["messages"] == [
        {"role": "user", "content": "hello agent"},
        {"role": "user", "content": "hello agent"},
    ]


def test_pre_roll_is_bounded_before_speech_start() -> None:
    transcriber = RecordingTranscriber()
    web_app = create_test_app(
        transcriber,
        FakeLanguageModel(),
        RecordingSpeechSynthesizer(),
        policy=SessionPolicy(silence_duration_ms=20, pre_roll_duration_ms=40),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "vad.started")
        websocket.send_json({"type": "session.stop"})

    assert transcriber.sessions[0].audio == [SILENCE_CHUNK, SPEECH_CHUNK]


def test_candidate_ready_before_commit_is_hidden_then_released() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello agent", "hello agent", None),),
        final_texts=("hello agent",),
    )
    language_model = PredictiveTrackingLanguageModel()
    prediction_source = DeterministicTurnPredictionSource(
        (
            PredictionDirective(p_user_speech=0.1, p_user_yield=0.7),
            PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),
        )
    )
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SPEECH_CHUNK)
        wait_until(lambda: language_model.completed_count == 1)

        assert sink.outputs == []
        assert transcriber.sessions[0].finish_count == 0

        websocket.send_bytes(SILENCE_CHUNK)
        events, audio_frame = receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        send_playback_started(websocket, 1)
        wait_until(
            lambda: sessions[0].generations[1].latency.first_browser_playback_ack is not None
        )
        latency = sessions[0].generations[1].latency
        websocket.send_json({"type": "session.stop"})

    assert audio_frame is None
    assert all(not event["type"].startswith("assistant.") for event in events)
    assert [conversation[-1].content for conversation in language_model.conversations] == [
        "hello agent"
    ]
    assert {output.generation_id for output in sink.outputs} == {1}
    assert any(isinstance(output, ReleasedTextDelta) for output in sink.outputs)
    assert any(isinstance(output, ReleasedAudioChunk) for output in sink.outputs)
    report = sessions[0].predictive_metrics.report()
    assert report.candidate_hit_rate == 1.0
    assert report.wasted_qwen_tokens == 0
    assert latency.first_endpoint is not None
    assert latency.speculation_start is not None
    assert latency.qwen_start is not None
    assert latency.qwen_first_complete_word is not None
    assert latency.tts_first_word is not None
    assert latency.tts_first_pcm is not None
    assert latency.turn_commitment is not None
    assert latency.asr_finalization is not None
    assert latency.candidate_resolution is not None
    assert latency.first_released_pcm is not None
    assert latency.first_browser_playback_ack is not None
    assert report.commit_to_first_played_audio_p50_ms is not None


def test_prediction_observed_before_resumed_speech_cannot_start_candidate() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", "hello", "hello", "hello", "hello"),),
        final_texts=("hello",),
    )
    language_model = PredictiveTrackingLanguageModel()
    prediction_source = DelayedTurnPredictionSource(
        PredictionDirective(p_user_speech=0.0, p_user_yield=0.95)
    )
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        policy=SessionPolicy(
            silence_duration_ms=200,
            pre_roll_duration_ms=20,
            vad_speculation_enabled=False,
        ),
        turn_prediction_source=prediction_source,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        assert prediction_source.started.wait(timeout=1)
        websocket.send_bytes(SPEECH_CHUNK)
        wait_until(lambda: transcriber.sessions[0].next_partial_index >= 3)
        prediction_source.release.set()
        assert prediction_source.completed.wait(timeout=1)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        wait_until(lambda: transcriber.sessions[0].next_partial_index >= 5)

        assert language_model.generation_started.is_set() is False
        websocket.send_json({"type": "session.stop"})

    assert sessions[0].generations == {}
    assert language_model.conversations == []


def test_prediction_beyond_policy_lag_cannot_start_candidate() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello",) * 8,),
        final_texts=("hello",),
    )
    language_model = PredictiveTrackingLanguageModel()
    prediction_source = DelayedTurnPredictionSource(
        PredictionDirective(p_user_speech=0.0, p_user_yield=0.95)
    )
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        policy=SessionPolicy(
            silence_duration_ms=500,
            pre_roll_duration_ms=20,
            vad_speculation_enabled=False,
            maximum_prediction_lag_ms=80,
        ),
        turn_prediction_source=prediction_source,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        assert prediction_source.started.wait(timeout=1)
        for _ in range(5):
            websocket.send_bytes(SILENCE_CHUNK)
        wait_until(lambda: transcriber.sessions[0].next_partial_index >= 7)
        prediction_source.release.set()
        assert prediction_source.completed.wait(timeout=1)
        wait_until(lambda: prediction_source.observation_count >= 6)
        websocket.send_bytes(SILENCE_CHUNK)
        wait_until(lambda: transcriber.sessions[0].next_partial_index >= 8)

        assert language_model.generation_started.is_set() is False
        websocket.send_json({"type": "session.stop"})

    assert sessions[0].generations == {}
    assert language_model.conversations == []


def test_first_vad_endpoint_speculates_during_commitment_silence() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello agent", None, None),),
        final_texts=("hello agent",),
    )
    language_model = PredictiveTrackingLanguageModel()
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        policy=SessionPolicy(
            silence_duration_ms=60,
            pre_roll_duration_ms=20,
            vad_speculation_enabled=True,
            vad_speculation_debounce_ms=40,
        ),
        playback_sink=sink,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        assert language_model.completed_count == 0
        assert transcriber.sessions[0].finish_count == 0

        websocket.send_bytes(SILENCE_CHUNK)
        wait_until(lambda: language_model.completed_count == 1)

        assert transcriber.sessions[0].finish_count == 0
        assert sink.outputs == []
        candidate = sessions[0].generations[1]
        assert candidate.causal_prediction is not None
        assert candidate.causal_prediction.stamp.source is CausalSource.SILERO_VAD

        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["generation_id"] == 1
    assert transcriber.sessions[0].finish_count == 1
    assert {output.generation_id for output in sink.outputs} == {1}
    report = sessions[0].predictive_metrics.report()
    assert report.candidate_hit_rate == 1.0
    assert report.hidden_qwen_tokens == 4
    assert report.hidden_tts_samples > 0


def test_clear_end_of_turn_can_create_and_commit_candidate_on_same_prediction() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", "hello"),),
        final_texts=("hello",),
    )
    language_model = PredictiveTrackingLanguageModel()
    prediction_source = DeterministicTurnPredictionSource(
        (PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),)
    )
    sink = InMemoryPlaybackSink()
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["generation_id"] == 1
    assert len(language_model.conversations) == 1
    assert {output.generation_id for output in sink.outputs} == {1}


def test_candidate_continues_streaming_after_commit() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello agent", "hello agent", None),),
        final_texts=("hello agent",),
    )
    language_model = PredictiveTrackingLanguageModel(block=True)
    prediction_source = DeterministicTurnPredictionSource(
        (
            PredictionDirective(p_user_speech=0.1, p_user_yield=0.7),
            PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),
        )
    )
    sink = InMemoryPlaybackSink()
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SPEECH_CHUNK)
        assert language_model.first_delta_produced.wait(timeout=1)
        assert sink.outputs == []

        websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioChunk) for output in sink.outputs))
        websocket.send_json({"type": "session.stop"})

    assert language_model.cancelled_count == 1
    assert language_model.overlapped is False
    assert {output.generation_id for output in sink.outputs} == {1}


def test_volatile_suffix_revision_does_not_invalidate_stable_candidate() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("book a", "book a", "book a table"),),
        final_texts=("book a!",),
    )
    language_model = PredictiveTrackingLanguageModel()
    prediction_source = DeterministicTurnPredictionSource(
        (
            PredictionDirective(p_user_speech=0.1, p_user_yield=0.7),
            PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),
        )
    )
    sink = InMemoryPlaybackSink()
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        websocket.send_json({"type": "session.stop"})

    assert len(language_model.conversations) == 1
    assert language_model.conversations[0][-1].content == "book a"
    assert {output.generation_id for output in sink.outputs} == {1}


def test_stable_prefix_revision_rejects_candidate_and_uses_new_generation_id() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("book a", "book a", "cancel it"),),
        final_texts=("cancel it",),
    )
    language_model = PredictiveTrackingLanguageModel()
    prediction_source = DeterministicTurnPredictionSource(
        (
            PredictionDirective(p_user_speech=0.1, p_user_yield=0.7),
            PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),
        )
    )
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        websocket.send_json({"type": "session.stop"})

    assert [conversation[-1].content for conversation in language_model.conversations] == [
        "book a",
        "cancel it",
    ]
    assert {output.generation_id for output in sink.outputs} == {2}
    invalidations = sessions[0].predictive_metrics.report().invalidations
    assert invalidations[0].reason is CandidateInvalidationReason.STABLE_PREFIX_REVISED


def test_speculative_tool_exchange_is_not_session_committed_after_revision() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("weather", "weather", "cancel it"),),
        final_texts=("cancel it",),
    )
    language_model = GenerationAwareToolLanguageModel()
    prediction_source = DeterministicTurnPredictionSource(
        (PredictionDirective(p_user_speech=0.1, p_user_yield=0.7),)
    )
    weather_handler = ControlledWeatherHandler()
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
        created_sessions=sessions,
        tool_executor=create_search_registry(weather_handler),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SPEECH_CHUNK)
        assert weather_handler.started.wait(timeout=1)
        weather_handler.release.set()
        assert language_model.old_continuation_started.wait(timeout=1)
        wait_until(
            lambda: (
                sessions[0].generations[1].tool_executions
                and sessions[0].generations[1].tool_executions[0].result_commit_status
                is ToolResultCommitStatus.GENERATION_LOCAL_COMMITTED
            )
        )
        assert sessions[0].model_context.snapshot() == ()
        assert sink.outputs == []

        websocket.send_bytes(SPEECH_CHUNK)
        wait_until(
            lambda: (
                sessions[0].generations[1].tool_executions
                and sessions[0].generations[1].tool_executions[0].result_commit_status
                is ToolResultCommitStatus.DISCARDED
            )
        )
        journal_entry = sessions[0].generations[1].tool_executions[0]
        assert journal_entry.lifecycle is ToolLifecycle.INVALIDATED
        assert journal_entry.invalidation_reason is ToolInvalidationReason.SPECULATIVE_INVALIDATION
        assert (
            journal_entry.candidate_invalidation_reason
            is CandidateInvalidationReason.STABLE_PREFIX_REVISED
        )
        assert sessions[0].model_context.snapshot() == ()
        assert sink.outputs == []
        websocket.send_json({"type": "session.stop"})


def test_user_resume_cancels_candidate_before_restart_without_qwen_overlap() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", "hello", "hello", None),),
        final_texts=("hello",),
    )
    language_model = PredictiveTrackingLanguageModel(block=True)
    prediction_source = DeterministicTurnPredictionSource(
        (
            PredictionDirective(p_user_speech=0.0, p_user_yield=0.7),
            PredictionDirective(p_user_speech=0.9, p_user_yield=0.1),
            PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),
        )
    )
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        assert language_model.first_delta_produced.wait(timeout=1)
        websocket.send_bytes(SPEECH_CHUNK)
        wait_until(lambda: language_model.cancelled_count == 1)
        websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioChunk) for output in sink.outputs))
        websocket.send_json({"type": "session.stop"})

    assert language_model.overlapped is False
    assert len(language_model.conversations) == 2
    assert {output.generation_id for output in sink.outputs} == {2}
    invalidation_reasons = {
        item.reason for item in sessions[0].predictive_metrics.report().invalidations
    }
    assert CandidateInvalidationReason.USER_ACTIVITY_RESUMED in invalidation_reasons


def test_long_continuation_pause_invalidates_then_restarts_at_commit() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", "hello", None, None),),
        final_texts=("hello",),
    )
    language_model = PredictiveTrackingLanguageModel()
    prediction_source = DeterministicTurnPredictionSource(
        (
            PredictionDirective(p_user_speech=0.0, p_user_yield=0.7),
            PredictionDirective(p_user_speech=0.9, p_user_yield=0.1),
            PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),
        )
    )
    sink = InMemoryPlaybackSink()
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        websocket.send_json({"type": "session.stop"})

    assert language_model.overlapped is False
    assert len(language_model.conversations) == 2
    assert {output.generation_id for output in sink.outputs} == {2}


def test_empty_final_transcript_discards_buffered_candidate() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", "hello"),),
        final_texts=("",),
    )
    language_model = PredictiveTrackingLanguageModel()
    prediction_source = DeterministicTurnPredictionSource(
        (PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),)
    )
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "transcript.final")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["text"] == ""
    assert sink.outputs == []
    report = sessions[0].predictive_metrics.report()
    assert report.invalidations[0].reason is CandidateInvalidationReason.EMPTY_FINAL_TRANSCRIPT


def test_session_shutdown_cancels_buffered_candidate_without_release() -> None:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", "hello"),),
        final_texts=("hello",),
    )
    language_model = PredictiveTrackingLanguageModel(block=True)
    prediction_source = DeterministicTurnPredictionSource(
        (PredictionDirective(p_user_speech=0.0, p_user_yield=0.7),)
    )
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        assert language_model.first_delta_produced.wait(timeout=1)
        websocket.send_json({"type": "session.stop"})

    assert sink.outputs == []
    assert language_model.cancelled_count == 1
    assert prediction_source.closed is True
    assert (
        sessions[0].predictive_metrics.report().invalidations[0].reason
        is CandidateInvalidationReason.SESSION_CANCELLED
    )


def test_predictive_candidate_measurably_hides_injected_qwen_latency() -> None:
    baseline_latency_ms = measure_commit_to_playback_latency(
        language_model=PredictiveTrackingLanguageModel(initial_delay_seconds=0.05),
        prediction_source=None,
        speculative=False,
    )
    predictive_latency_ms = measure_commit_to_playback_latency(
        language_model=PredictiveTrackingLanguageModel(initial_delay_seconds=0.05),
        prediction_source=DeterministicTurnPredictionSource(
            (
                PredictionDirective(p_user_speech=0.0, p_user_yield=0.7),
                PredictionDirective(p_user_speech=0.0, p_user_yield=0.95),
            )
        ),
        speculative=True,
    )

    assert baseline_latency_ms - predictive_latency_ms >= 30.0


def measure_commit_to_playback_latency(
    language_model: PredictiveTrackingLanguageModel,
    prediction_source: TurnPredictionSource | None,
    speculative: bool,
) -> float:
    transcriber = ScriptedTranscriber(
        partials_by_turn=(("hello", "hello", None),),
        final_texts=("hello",),
    )
    sink = InMemoryPlaybackSink()
    sessions: list[VoiceSession] = []
    web_app = create_test_app(
        transcriber,
        language_model,
        RecordingSpeechSynthesizer(),
        turn_prediction_source=prediction_source,
        playback_sink=sink,
        created_sessions=sessions,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SPEECH_CHUNK)
        websocket.send_bytes(SPEECH_CHUNK if speculative else SILENCE_CHUNK)
        if speculative:
            wait_until(lambda: language_model.completed_count == 1)
            websocket.send_bytes(SILENCE_CHUNK)
        else:
            websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "llm.history")
        wait_until(lambda: any(isinstance(output, ReleasedAudioEnd) for output in sink.outputs))
        send_playback_started(websocket, 1)
        wait_until(
            lambda: (
                sessions[0].predictive_metrics.report().commit_to_first_played_audio_p50_ms
                is not None
            )
        )
        measured_latency_ms = (
            sessions[0].predictive_metrics.report().commit_to_first_played_audio_p50_ms
        )
        websocket.send_json({"type": "session.stop"})

    assert measured_latency_ms is not None
    return measured_latency_ms


def create_test_app(
    transcriber: Transcriber,
    language_model: LanguageModel,
    speech_synthesizer: SpeechSynthesizer,
    policy: SessionPolicy = DEFAULT_TEST_POLICY,
    speech_detector: SpeechDetector = DEFAULT_TEST_SPEECH_DETECTOR,
    turn_prediction_source: TurnPredictionSource | None = None,
    playback_sink: PlaybackSink | None = None,
    created_sessions: list[VoiceSession] | None = None,
    tool_executor: ToolExecutor | None = None,
) -> FastAPI:
    web_app = FastAPI()

    @web_app.websocket("/session")
    async def endpoint(websocket: WebSocket) -> None:
        turn_prediction_provider = (
            None
            if turn_prediction_source is None
            else SingleSessionTurnPredictionProvider(turn_prediction_source)
        )
        speech_understanding_provider = CompositeSpeechUnderstandingProvider(
            transcriber=transcriber,
            turn_prediction_provider=turn_prediction_provider,
            asr_model_name="test-asr",
            asr_model_revision="1",
        )
        session = VoiceSession(
            websocket=websocket,
            speech_detector=speech_detector,
            speech_understanding_provider=speech_understanding_provider,
            language_model=language_model,
            speech_synthesizer=speech_synthesizer,
            policy=policy,
            tool_executor=tool_executor or create_search_registry(UnconfiguredTestSearchHandler()),
            playback_sink=playback_sink,
        )
        if created_sessions is not None:
            created_sessions.append(session)
        await session.run()

    return web_app


def send_turn(websocket: WebSocketTestSession) -> None:
    websocket.send_bytes(SPEECH_CHUNK)
    websocket.send_bytes(SILENCE_CHUNK)
    websocket.send_bytes(SILENCE_CHUNK)


def send_playback_started(
    websocket: WebSocketTestSession,
    generation_id: int,
) -> None:
    websocket.send_text(
        PlaybackStartedEvent(
            generation_id=generation_id,
            browser_monotonic_time_ns=time.perf_counter_ns(),
            rendered_output_sample_position=1,
            source_sample_position=1,
            output_sample_rate=48_000,
        ).model_dump_json()
    )


def send_playback_complete(
    websocket: WebSocketTestSession,
    generation_id: int,
) -> None:
    websocket.send_text(
        PlaybackCompleteEvent(
            generation_id=generation_id,
            browser_monotonic_time_ns=time.perf_counter_ns(),
            rendered_output_sample_position=1_000,
            source_sample_position=500,
            output_sample_rate=48_000,
        ).model_dump_json()
    )


def send_playback_progress(
    websocket: WebSocketTestSession,
    generation_id: int,
    text_offset: int,
    boundary_start_sample: int,
    played_sample_count: int,
) -> None:
    websocket.send_text(
        PlaybackProgressEvent(
            generation_id=generation_id,
            text_offset=text_offset,
            boundary_start_sample=boundary_start_sample,
            played_sample_count=played_sample_count,
            browser_monotonic_time_ns=time.perf_counter_ns(),
            rendered_output_sample_position=played_sample_count * 2,
            output_sample_rate=48_000,
        ).model_dump_json()
    )


def send_playback_stopped(
    websocket: WebSocketTestSession,
    generation_id: int,
    text_offset: int,
    played_sample_count: int,
) -> None:
    websocket.send_text(
        PlaybackStoppedEvent(
            generation_id=generation_id,
            text_offset=text_offset,
            played_sample_count=played_sample_count,
            browser_monotonic_time_ns=time.perf_counter_ns(),
            rendered_output_sample_position=played_sample_count * 2,
            output_sample_rate=48_000,
        ).model_dump_json()
    )


def receive_playback_command(
    websocket: WebSocketTestSession,
) -> PlaybackCommandEvent:
    events, _ = receive_until(websocket, "playback.command")
    return PlaybackCommandEvent.model_validate(events[-1])


def send_playback_command_acknowledgement(
    websocket: WebSocketTestSession,
    command: PlaybackCommandEvent,
    resulting_state: PlaybackState,
    pause_result: PlaybackPauseResult,
    source_sample_position: int,
) -> None:
    websocket.send_text(
        PlaybackCommandAcknowledgementEvent(
            command_id=command.command_id,
            generation_id=command.generation_id,
            action=command.action,
            stream_epoch=command.stream_epoch,
            turn_epoch=command.turn_epoch,
            resulting_state=resulting_state,
            browser_monotonic_time_ns=time.perf_counter_ns(),
            rendered_output_sample_position=source_sample_position * 2,
            source_sample_position=source_sample_position,
            output_sample_rate=48_000,
            pause_result=pause_result,
            current_gain=0.1258925,
            gain_ramp_complete=True,
            queued_source_sample_count=100,
            discarded_source_sample_count=0,
            replayed_source_sample_count=0,
            skipped_source_sample_count=0,
            resume_rejected=False,
        ).model_dump_json()
    )


def receive_until(
    websocket: WebSocketTestSession,
    terminal_event_type: str,
) -> tuple[list[dict[str, object]], bytes | None]:
    events: list[dict[str, object]] = []
    audio_frame: bytes | None = None
    while not events or events[-1]["type"] != terminal_event_type:
        message = websocket.receive()
        if message.get("bytes") is not None:
            audio_frame = message["bytes"]
        elif message.get("text") is not None:
            events.append(json.loads(message["text"]))
    return events, audio_frame


def wait_until(condition: Callable[[], bool], timeout_seconds: float = 1.0) -> None:
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for the test condition.")
