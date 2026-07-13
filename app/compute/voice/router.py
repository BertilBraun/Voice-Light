from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
from collections.abc import Coroutine

from fastapi import WebSocket, WebSocketDisconnect

from app.compute.runtime import ComputeRuntime
from app.compute.telemetry import RequestIdScope
from app.compute.voice.interfaces import TranscriptionSession
from app.shared.compute_api import (
    AsrAudioCommand,
    AsrCloseCommand,
    AsrFinalEvent,
    AsrFinishCommand,
    AsrPartialEvent,
    AsrStartCommand,
    ComputeReadyEvent,
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
    voice_client_event_adapter,
)

logger = logging.getLogger(__name__)


class VoiceComputeSession:
    def __init__(self, websocket: WebSocket, runtime: ComputeRuntime, request_id: str) -> None:
        self.websocket = websocket
        self.runtime = runtime
        self.request_id = request_id
        self.send_lock = asyncio.Lock()
        self.transcriptions: dict[str, TranscriptionSession] = {}
        self.operation_tasks: dict[str, asyncio.Task[None]] = {}

    async def run(self) -> None:
        synthesizer = self.runtime.require_speech_synthesizer()
        await self.websocket.accept()
        await self._send(ComputeReadyEvent(output_sample_rate=synthesizer.sample_rate))
        logger.info("voice compute session connected")
        try:
            while True:
                command = voice_client_event_adapter.validate_json(
                    await self.websocket.receive_text()
                )
                await self._handle(command)
        except WebSocketDisconnect:
            logger.info("voice compute session disconnected")
        finally:
            await self._close()

    async def _handle(self, command: VoiceClientEvent) -> None:
        try:
            match command:
                case AsrStartCommand():
                    await self._start_asr(command)
                case AsrAudioCommand():
                    await self._add_asr_audio(command)
                case AsrFinishCommand():
                    await self._finish_asr(command)
                case AsrCloseCommand():
                    await self._close_asr(command)
                case LanguageModelGenerateCommand():
                    self._start_operation(
                        command.operation_id,
                        self._generate_language(command),
                    )
                case SpeechSynthesizeCommand():
                    self._start_operation(
                        command.operation_id,
                        self._synthesize_speech(command),
                    )
                case OperationCancelCommand():
                    await self._cancel_operation(command.operation_id)
        except Exception as error:
            logger.exception("voice command failed: %s", command.type)
            await self._send(
                OperationErrorEvent(
                    operation_id=command.operation_id,
                    message=str(error),
                )
            )

    async def _start_asr(self, command: AsrStartCommand) -> None:
        if command.operation_id in self.transcriptions:
            raise ValueError("ASR operation already exists.")
        transcriber = self.runtime.require_streaming_asr()
        self.transcriptions[command.operation_id] = transcriber.start_session()

    async def _add_asr_audio(self, command: AsrAudioCommand) -> None:
        transcription = self._transcription(command.operation_id)
        try:
            pcm_bytes = base64.b64decode(command.audio_base64, validate=True)
        except ValueError as error:
            raise ValueError("Invalid ASR audio payload.") from error
        partial_text = await transcription.add_audio(pcm_bytes)
        if partial_text:
            await self._send(AsrPartialEvent(operation_id=command.operation_id, text=partial_text))

    async def _finish_asr(self, command: AsrFinishCommand) -> None:
        transcription = self._transcription(command.operation_id)
        started = time.perf_counter()
        text = await transcription.finish()
        self.transcriptions.pop(command.operation_id)
        elapsed = time.perf_counter() - started
        logger.info("streaming ASR finalized in %.3f seconds", elapsed)
        await self._send(
            AsrFinalEvent(
                operation_id=command.operation_id,
                text=text,
                inference_time_seconds=elapsed,
            )
        )

    async def _close_asr(self, command: AsrCloseCommand) -> None:
        transcription = self.transcriptions.pop(command.operation_id, None)
        if transcription is not None:
            await transcription.close()

    async def _generate_language(self, command: LanguageModelGenerateCommand) -> None:
        language_model = self.runtime.require_language_model()
        started = time.perf_counter()
        async for text_delta in language_model.stream_response(command.conversation):
            await self._send(
                LanguageModelDeltaEvent(
                    operation_id=command.operation_id,
                    text=text_delta,
                )
            )
        elapsed = time.perf_counter() - started
        logger.info("language generation completed in %.3f seconds", elapsed)
        await self._send(
            LanguageModelEndEvent(
                operation_id=command.operation_id,
                inference_time_seconds=elapsed,
            )
        )

    async def _synthesize_speech(self, command: SpeechSynthesizeCommand) -> None:
        synthesizer = self.runtime.require_speech_synthesizer()
        started = time.perf_counter()
        sample_count = 0
        await self._send(SpeechStartEvent(operation_id=command.operation_id))
        async for pcm_bytes in synthesizer.stream_audio(command.text):
            sample_count += len(pcm_bytes) // 2
            await self._send(
                SpeechAudioEvent(
                    operation_id=command.operation_id,
                    audio_base64=base64.b64encode(pcm_bytes).decode("ascii"),
                )
            )
        elapsed = time.perf_counter() - started
        audio_duration_seconds = sample_count / synthesizer.sample_rate
        logger.info(
            "speech synthesis completed in %.3f seconds for %.3f seconds of audio",
            elapsed,
            audio_duration_seconds,
        )
        await self._send(
            SpeechEndEvent(
                operation_id=command.operation_id,
                inference_time_seconds=elapsed,
                audio_duration_seconds=audio_duration_seconds,
            )
        )

    def _start_operation(
        self,
        operation_id: str,
        operation: Coroutine[None, None, None],
    ) -> None:
        existing_task = self.operation_tasks.get(operation_id)
        if existing_task is not None and not existing_task.done():
            operation.close()
            raise ValueError("Operation already exists.")
        task = asyncio.create_task(self._run_operation(operation_id, operation))
        self.operation_tasks[operation_id] = task

    async def _run_operation(
        self,
        operation_id: str,
        operation: Coroutine[None, None, None],
    ) -> None:
        try:
            await operation
        except asyncio.CancelledError:
            logger.info("operation cancelled: %s", operation_id)
            raise
        except Exception as error:
            logger.exception("operation failed: %s", operation_id)
            await self._send(OperationErrorEvent(operation_id=operation_id, message=str(error)))

    async def _cancel_operation(self, operation_id: str) -> None:
        task = self.operation_tasks.get(operation_id)
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _transcription(self, operation_id: str) -> TranscriptionSession:
        transcription = self.transcriptions.get(operation_id)
        if transcription is None:
            raise ValueError("ASR operation does not exist.")
        return transcription

    async def _send(self, event: VoiceServerEvent) -> None:
        async with self.send_lock:
            await self.websocket.send_text(event.model_dump_json())

    async def _close(self) -> None:
        for task in self.operation_tasks.values():
            if not task.done():
                task.cancel()
        for task in self.operation_tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for transcription in self.transcriptions.values():
            await transcription.close()
        self.transcriptions.clear()


async def run_voice_compute_session(
    websocket: WebSocket,
    runtime: ComputeRuntime,
    request_id: str,
) -> None:
    with RequestIdScope(request_id):
        session = VoiceComputeSession(
            websocket=websocket,
            runtime=runtime,
            request_id=request_id,
        )
        await session.run()
