from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TypeVar

from app.compute.asr.models.registry import AsrModelCache
from app.compute.voice.admission import SingleVoiceSessionAdmission, VoiceSessionLease
from app.compute.voice.kyutai_tts import KyutaiSpeechSynthesizer
from app.compute.voice.models import TransformersLanguageModel
from app.compute.voice.nemotron_client import NemotronStreamingTranscriber
from app.shared.compute_api import ModelStage, ModelStageStatus

logger = logging.getLogger(__name__)
ModelType = TypeVar("ModelType")


@dataclass
class MutableModelStage:
    name: str
    status: ModelStageStatus = ModelStageStatus.PENDING
    load_time_seconds: float | None = None
    error: str | None = None

    def snapshot(self) -> ModelStage:
        return ModelStage(
            name=self.name,
            status=self.status,
            load_time_seconds=self.load_time_seconds,
            error=self.error,
        )


class ComputeRuntime:
    def __init__(self) -> None:
        self.streaming_asr_stage = MutableModelStage("streaming_asr")
        self.language_model_stage = MutableModelStage("language_model")
        self.speech_synthesis_stage = MutableModelStage("speech_synthesis")
        self.batch_asr_models = AsrModelCache()
        self.voice_session_admission = SingleVoiceSessionAdmission()
        self.streaming_asr: NemotronStreamingTranscriber | None = None
        self.language_model: TransformersLanguageModel | None = None
        self.speech_synthesizer: KyutaiSpeechSynthesizer | None = None
        self.loading_task: asyncio.Task[None] | None = None

    @property
    def ready(self) -> bool:
        return all(stage.status is ModelStageStatus.READY for stage in self._stages())

    def stages(self) -> tuple[ModelStage, ...]:
        return tuple(stage.snapshot() for stage in self._stages())

    def try_admit_voice_session(self, request_id: str) -> VoiceSessionLease | None:
        return self.voice_session_admission.try_acquire(request_id)

    def release_voice_session(self, lease: VoiceSessionLease) -> None:
        self.voice_session_admission.release(lease)

    def start_loading(self) -> None:
        assert self.loading_task is None
        self.loading_task = asyncio.create_task(self._load_models())

    async def shutdown(self) -> None:
        if self.loading_task is not None and not self.loading_task.done():
            self.loading_task.cancel()
            try:
                await self.loading_task
            except asyncio.CancelledError:
                pass
        if self.streaming_asr is not None:
            await asyncio.to_thread(self.streaming_asr.close)
        if self.speech_synthesizer is not None:
            await asyncio.to_thread(self.speech_synthesizer.close)
        logger.info("compute runtime shutdown complete")

    def require_streaming_asr(self) -> NemotronStreamingTranscriber:
        if self.streaming_asr is None:
            raise RuntimeError("Streaming ASR is not ready.")
        return self.streaming_asr

    def require_language_model(self) -> TransformersLanguageModel:
        if self.language_model is None:
            raise RuntimeError("Language model is not ready.")
        return self.language_model

    def require_speech_synthesizer(self) -> KyutaiSpeechSynthesizer:
        if self.speech_synthesizer is None:
            raise RuntimeError("Speech synthesizer is not ready.")
        return self.speech_synthesizer

    async def _load_models(self) -> None:
        await self._load_streaming_asr()
        await self._load_language_model()
        await self._load_speech_synthesizer()
        logger.info("all required compute models ready")

    async def _load_streaming_asr(self) -> None:
        model = await self._timed_load(
            self.streaming_asr_stage,
            NemotronStreamingTranscriber,
        )
        if model is not None:
            self.streaming_asr = model

    async def _load_language_model(self) -> None:
        model = await self._timed_load(
            self.language_model_stage,
            TransformersLanguageModel,
        )
        if model is not None:
            self.language_model = model

    async def _load_speech_synthesizer(self) -> None:
        model = await self._timed_load(
            self.speech_synthesis_stage,
            KyutaiSpeechSynthesizer,
        )
        if model is not None:
            self.speech_synthesizer = model

    async def _timed_load(
        self,
        stage: MutableModelStage,
        model_type: type[ModelType],
    ) -> ModelType | None:
        stage.status = ModelStageStatus.LOADING
        logger.info("model load started: %s", stage.name)
        started = time.perf_counter()
        try:
            model = await asyncio.to_thread(model_type)
        except Exception as error:
            stage.status = ModelStageStatus.FAILED
            stage.error = str(error)
            stage.load_time_seconds = time.perf_counter() - started
            logger.exception("model load failed: %s", stage.name)
            return None
        stage.status = ModelStageStatus.READY
        stage.load_time_seconds = time.perf_counter() - started
        logger.info(
            "model load completed: %s in %.3f seconds",
            stage.name,
            stage.load_time_seconds,
        )
        return model

    def _stages(self) -> tuple[MutableModelStage, ...]:
        return (
            self.streaming_asr_stage,
            self.language_model_stage,
            self.speech_synthesis_stage,
        )
