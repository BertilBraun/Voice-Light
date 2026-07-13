from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator
from typing import Protocol
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect

from app.local.voice.interfaces import TranscriptionSession
from app.shared.compute_api import (
    AsrAudioCommand,
    AsrCloseCommand,
    AsrFinalEvent,
    AsrFinishCommand,
    AsrPartialEvent,
    AsrStartCommand,
    ComputeReadyEvent,
    ConversationMessage,
    ConversationRole,
    LanguageModelDeltaEvent,
    LanguageModelEndEvent,
    LanguageModelGenerateCommand,
    OperationCancelCommand,
    OperationErrorEvent,
    SpeechAudioEvent,
    SpeechEndEvent,
    SpeechStartEvent,
    SpeechSynthesizeCommand,
    VoiceClientEvent,
    VoiceServerEvent,
    voice_server_event_adapter,
)


class ComputeVoiceChannel(Protocol):
    output_sample_rate: int | None

    def register_operation(self, operation_id: str) -> None: ...

    def unregister_operation(self, operation_id: str) -> None: ...

    async def send(self, event: VoiceClientEvent) -> None: ...

    async def next_event(self, operation_id: str) -> VoiceServerEvent: ...

    def pending_event(self, operation_id: str) -> VoiceServerEvent | None: ...


class RemoteComputeVoiceChannel:
    def __init__(self, base_url: str, token: str) -> None:
        if not base_url:
            raise ValueError("VOICE_LIGHT_COMPUTE_URL is required for voice sessions.")
        if not token:
            raise ValueError("VOICE_LIGHT_COMPUTE_TOKEN is required for voice sessions.")
        self.websocket_url = compute_websocket_url(base_url)
        self.token = token
        self.websocket: ClientConnection | None = None
        self.receive_task: asyncio.Task[None] | None = None
        self.send_lock = asyncio.Lock()
        self.operation_queues: dict[
            str,
            asyncio.Queue[VoiceServerEvent | Exception],
        ] = {}
        self.output_sample_rate: int | None = None

    async def connect(self) -> None:
        self.websocket = await connect(
            self.websocket_url,
            additional_headers={
                "Authorization": f"Bearer {self.token}",
                "X-Request-ID": str(uuid4()),
            },
            max_size=8 * 1024 * 1024,
        )
        payload = await self.websocket.recv()
        if not isinstance(payload, str):
            raise RuntimeError("Compute backend sent an invalid readiness message.")
        event = voice_server_event_adapter.validate_json(payload)
        if not isinstance(event, ComputeReadyEvent):
            raise RuntimeError("Compute backend did not send readiness information.")
        self.output_sample_rate = event.output_sample_rate
        self.receive_task = asyncio.create_task(self._receive_loop())

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()
        if self.receive_task is not None:
            self.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.receive_task

    def register_operation(self, operation_id: str) -> None:
        if operation_id in self.operation_queues:
            raise ValueError("Compute operation already exists.")
        self.operation_queues[operation_id] = asyncio.Queue()

    def unregister_operation(self, operation_id: str) -> None:
        self.operation_queues.pop(operation_id, None)

    async def send(self, event: VoiceClientEvent) -> None:
        websocket = self._websocket()
        async with self.send_lock:
            await websocket.send(event.model_dump_json())

    async def next_event(self, operation_id: str) -> VoiceServerEvent:
        queue = self._operation_queue(operation_id)
        event = await queue.get()
        if isinstance(event, Exception):
            raise RuntimeError("Compute voice connection failed.") from event
        if isinstance(event, OperationErrorEvent):
            raise RuntimeError(event.message)
        return event

    def pending_event(self, operation_id: str) -> VoiceServerEvent | None:
        queue = self._operation_queue(operation_id)
        try:
            event = queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if isinstance(event, Exception):
            raise RuntimeError("Compute voice connection failed.") from event
        if isinstance(event, OperationErrorEvent):
            raise RuntimeError(event.message)
        return event

    async def _receive_loop(self) -> None:
        websocket = self._websocket()
        try:
            async for payload in websocket:
                if not isinstance(payload, str):
                    raise RuntimeError("Compute backend sent an unexpected binary message.")
                event = voice_server_event_adapter.validate_json(payload)
                if isinstance(event, ComputeReadyEvent):
                    raise RuntimeError("Compute backend sent duplicate readiness information.")
                queue = self.operation_queues.get(event.operation_id)
                if queue is not None:
                    await queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            for queue in self.operation_queues.values():
                await queue.put(error)

    def _websocket(self) -> ClientConnection:
        if self.websocket is None:
            raise RuntimeError("Compute voice channel is not connected.")
        return self.websocket

    def _operation_queue(
        self,
        operation_id: str,
    ) -> asyncio.Queue[VoiceServerEvent | Exception]:
        queue = self.operation_queues.get(operation_id)
        if queue is None:
            raise RuntimeError("Compute operation is not registered.")
        return queue


class RemoteStreamingTranscriber:
    def __init__(self, channel: ComputeVoiceChannel) -> None:
        self.channel = channel

    def start_session(self) -> TranscriptionSession:
        return RemoteTranscriptionSession(channel=self.channel)


class RemoteTranscriptionSession:
    def __init__(self, channel: ComputeVoiceChannel) -> None:
        self.channel = channel
        self.operation_id = str(uuid4())
        self.channel.register_operation(self.operation_id)
        self.started = False
        self.finished = False

    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        if self.finished:
            raise RuntimeError("Cannot add audio to a finished transcription.")
        await self._ensure_started()
        await self.channel.send(
            AsrAudioCommand(
                operation_id=self.operation_id,
                audio_base64=base64.b64encode(pcm_bytes).decode("ascii"),
            )
        )
        await asyncio.sleep(0)
        latest_partial: str | None = None
        while (event := self.channel.pending_event(self.operation_id)) is not None:
            match event:
                case AsrPartialEvent():
                    latest_partial = event.text
                case _:
                    raise RuntimeError("Compute backend sent an unexpected ASR event.")
        return latest_partial

    async def finish(self) -> str:
        if self.finished:
            raise RuntimeError("Cannot finish a transcription twice.")
        self.finished = True
        if not self.started:
            self.channel.unregister_operation(self.operation_id)
            return ""
        await self.channel.send(AsrFinishCommand(operation_id=self.operation_id))
        try:
            while True:
                event = await self.channel.next_event(self.operation_id)
                match event:
                    case AsrPartialEvent():
                        continue
                    case AsrFinalEvent():
                        return event.text
                    case _:
                        raise RuntimeError("Compute backend sent an unexpected ASR event.")
        finally:
            self.channel.unregister_operation(self.operation_id)

    async def close(self) -> None:
        if self.finished:
            return
        self.finished = True
        if self.started:
            await self.channel.send(AsrCloseCommand(operation_id=self.operation_id))
        self.channel.unregister_operation(self.operation_id)

    async def _ensure_started(self) -> None:
        if self.started:
            return
        self.started = True
        await self.channel.send(AsrStartCommand(operation_id=self.operation_id))


class RemoteLanguageModel:
    def __init__(self, channel: ComputeVoiceChannel) -> None:
        self.channel = channel

    async def stream_response(
        self,
        conversation: tuple[tuple[str, str], ...],
    ) -> AsyncIterator[str]:
        operation_id = str(uuid4())
        self.channel.register_operation(operation_id)
        completed = False
        messages = tuple(
            ConversationMessage(role=ConversationRole(role), content=content)
            for role, content in conversation
        )
        await self.channel.send(
            LanguageModelGenerateCommand(
                operation_id=operation_id,
                conversation=messages,
            )
        )
        try:
            while True:
                event = await self.channel.next_event(operation_id)
                match event:
                    case LanguageModelDeltaEvent():
                        yield event.text
                    case LanguageModelEndEvent():
                        completed = True
                        return
                    case _:
                        raise RuntimeError("Compute backend sent an unexpected LLM event.")
        finally:
            if not completed:
                await self.channel.send(OperationCancelCommand(operation_id=operation_id))
            self.channel.unregister_operation(operation_id)


class RemoteSpeechSynthesizer:
    def __init__(self, channel: ComputeVoiceChannel) -> None:
        if channel.output_sample_rate is None:
            raise RuntimeError("Compute voice channel is not ready.")
        self.channel = channel
        self.output_sample_rate = channel.output_sample_rate

    @property
    def sample_rate(self) -> int:
        return self.output_sample_rate

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        operation_id = str(uuid4())
        self.channel.register_operation(operation_id)
        completed = False
        await self.channel.send(SpeechSynthesizeCommand(operation_id=operation_id, text=text))
        try:
            while True:
                event = await self.channel.next_event(operation_id)
                match event:
                    case SpeechStartEvent():
                        continue
                    case SpeechAudioEvent():
                        yield base64.b64decode(event.audio_base64, validate=True)
                    case SpeechEndEvent():
                        completed = True
                        return
                    case _:
                        raise RuntimeError("Compute backend sent an unexpected TTS event.")
        finally:
            if not completed:
                await self.channel.send(OperationCancelCommand(operation_id=operation_id))
            self.channel.unregister_operation(operation_id)


def compute_websocket_url(base_url: str) -> str:
    parsed_url = urlparse(base_url)
    match parsed_url.scheme:
        case "http":
            websocket_scheme = "ws"
        case "https":
            websocket_scheme = "wss"
        case _:
            raise ValueError("VOICE_LIGHT_COMPUTE_URL must use HTTP or HTTPS.")
    if not parsed_url.netloc:
        raise ValueError("VOICE_LIGHT_COMPUTE_URL must include a host.")
    base_path = parsed_url.path.rstrip("/")
    return urlunparse(
        parsed_url._replace(
            scheme=websocket_scheme,
            path=f"{base_path}/v1/voice",
            params="",
            query="",
            fragment="",
        )
    )
