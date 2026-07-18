from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from app.compute.asr.models.registry import AsrModelCache
from app.compute.config import VoiceStackSettings
from app.compute.voice.admission import SingleVoiceSessionAdmission, VoiceSessionLease
from app.compute.voice.interfaces import SpeechSynthesizer, SpeechUnderstandingProvider
from app.compute.voice.model_constants import (
    NEMOTRON_ASR_MODEL_NAME,
    NEMOTRON_ASR_MODEL_REVISION,
)
from app.compute.voice.models import TransformersLanguageModel, TransformersTextGenerator
from app.compute.voice.nemotron_client import NemotronStreamingTranscriber
from app.compute.voice.search import SearchProvider, create_search_provider
from app.compute.voice.speech_detection import SileroSpeechDetectorFactory
from app.compute.voice.speech_understanding import CompositeSpeechUnderstandingProvider
from app.compute.voice.tts_selection import create_speech_synthesizer
from app.shared.compute_api import ModelStage, ModelStageStatus

logger = logging.getLogger(__name__)
ModelType = TypeVar("ModelType")
LoaderParameters = ParamSpec("LoaderParameters")


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
    def __init__(self, voice_stack_settings: VoiceStackSettings | None) -> None:
        self.voice_stack_settings = voice_stack_settings
        self.streaming_asr_stage = MutableModelStage("streaming_asr")
        self.speech_detection_stage = MutableModelStage("speech_detection")
        self.language_model_stage = MutableModelStage("language_model")
        self.search_summarizer_stage = MutableModelStage("search_summarizer")
        self.speech_synthesis_stage = MutableModelStage("speech_synthesis")
        self.batch_asr_models = AsrModelCache()
        self.voice_session_admission = SingleVoiceSessionAdmission()
        self.speech_understanding_provider: SpeechUnderstandingProvider | None = None
        self.speech_detector_factory: SileroSpeechDetectorFactory | None = None
        self.qwen_inference_lock = asyncio.Lock()
        self.language_model: TransformersLanguageModel | None = None
        self.search_text_generator: TransformersTextGenerator | None = None
        self.speech_synthesizer: SpeechSynthesizer | None = None
        self.search_provider: SearchProvider | None = (
            None
            if voice_stack_settings is None
            else create_search_provider(voice_stack_settings.search)
        )
        self.loading_task: asyncio.Task[None] | None = None

    @property
    def ready(self) -> bool:
        return not self.voice_enabled or self.voice_ready

    @property
    def voice_enabled(self) -> bool:
        return self.voice_stack_settings is not None

    @property
    def voice_ready(self) -> bool:
        return self.voice_enabled and all(
            stage.status is ModelStageStatus.READY for stage in self._voice_stages()
        )

    def stages(self) -> tuple[ModelStage, ...]:
        if self.voice_stack_settings is None:
            return ()
        return tuple(stage.snapshot() for stage in self._voice_stages())

    def try_admit_voice_session(self, request_id: str) -> VoiceSessionLease | None:
        return self.voice_session_admission.try_acquire(request_id)

    def release_voice_session(self, lease: VoiceSessionLease) -> None:
        self.voice_session_admission.release(lease)

    def start_loading(self) -> None:
        assert self.loading_task is None
        if self.voice_stack_settings is None:
            logger.info("voice stack disabled; batch ASR is ready")
            return
        self.loading_task = asyncio.create_task(self._load_models())

    async def shutdown(self) -> None:
        if self.loading_task is not None and not self.loading_task.done():
            self.loading_task.cancel()
            try:
                await self.loading_task
            except asyncio.CancelledError:
                pass
        if self.speech_understanding_provider is not None:
            await asyncio.to_thread(self.speech_understanding_provider.close)
        if self.language_model is not None:
            await asyncio.to_thread(self.language_model.close)
        if self.search_text_generator is not None:
            await asyncio.to_thread(self.search_text_generator.close)
        if self.speech_synthesizer is not None:
            await asyncio.to_thread(self.speech_synthesizer.close)
        if self.search_provider is not None:
            await self.search_provider.close()
        logger.info("compute runtime shutdown complete")

    def require_speech_understanding_provider(self) -> SpeechUnderstandingProvider:
        if self.speech_understanding_provider is None:
            raise RuntimeError("Speech understanding is not ready.")
        return self.speech_understanding_provider

    def require_speech_detector_factory(self) -> SileroSpeechDetectorFactory:
        if self.speech_detector_factory is None:
            raise RuntimeError("Speech detector is not ready.")
        return self.speech_detector_factory

    def require_language_model(self) -> TransformersLanguageModel:
        if self.language_model is None:
            raise RuntimeError("Language model is not ready.")
        return self.language_model

    def require_search_text_generator(self) -> TransformersTextGenerator:
        if self.search_text_generator is None:
            raise RuntimeError("Search summarizer is not ready.")
        return self.search_text_generator

    def require_speech_synthesizer(self) -> SpeechSynthesizer:
        if self.speech_synthesizer is None:
            raise RuntimeError("Speech synthesizer is not ready.")
        return self.speech_synthesizer

    def require_search_provider(self) -> SearchProvider:
        if self.search_provider is None:
            raise RuntimeError("Search provider is not ready.")
        return self.search_provider

    async def _load_models(self) -> None:
        await self._load_speech_detector()
        await self._load_streaming_asr()
        await self._load_language_model()
        await self._load_search_text_generator()
        await self._load_speech_synthesizer()
        logger.info("all required compute models ready")

    async def _load_speech_detector(self) -> None:
        factory = await self._timed_load(
            self.speech_detection_stage,
            SileroSpeechDetectorFactory,
        )
        if factory is not None:
            self.speech_detector_factory = factory

    async def _load_streaming_asr(self) -> None:
        transcriber = await self._timed_load(
            self.streaming_asr_stage,
            NemotronStreamingTranscriber,
        )
        if transcriber is not None:
            self.speech_understanding_provider = CompositeSpeechUnderstandingProvider(
                transcriber=transcriber,
                turn_prediction_provider=None,
                asr_model_name=NEMOTRON_ASR_MODEL_NAME,
                asr_model_revision=NEMOTRON_ASR_MODEL_REVISION,
            )

    async def _load_language_model(self) -> None:
        model = await self._timed_load(
            self.language_model_stage,
            TransformersLanguageModel,
            self.qwen_inference_lock,
        )
        if model is not None:
            self.language_model = model

    async def _load_search_text_generator(self) -> None:
        generator = await self._timed_load(
            self.search_summarizer_stage,
            TransformersTextGenerator,
            self.qwen_inference_lock,
        )
        if generator is not None:
            self.search_text_generator = generator

    async def _load_speech_synthesizer(self) -> None:
        assert self.voice_stack_settings is not None
        model = await self._timed_load(
            self.speech_synthesis_stage,
            create_speech_synthesizer,
            self.voice_stack_settings.speech_synthesis,
        )
        if model is not None:
            self.speech_synthesizer = model

    async def _timed_load(
        self,
        stage: MutableModelStage,
        loader: Callable[LoaderParameters, ModelType],
        *loader_arguments: LoaderParameters.args,
    ) -> ModelType | None:
        stage.status = ModelStageStatus.LOADING
        logger.info("model load started: %s", stage.name)
        started = time.perf_counter()
        try:
            model = await asyncio.to_thread(loader, *loader_arguments)
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

    def _voice_stages(self) -> tuple[MutableModelStage, ...]:
        return (
            self.speech_detection_stage,
            self.streaming_asr_stage,
            self.language_model_stage,
            self.search_summarizer_stage,
            self.speech_synthesis_stage,
        )
