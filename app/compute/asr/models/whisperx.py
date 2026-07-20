from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from app.compute.asr.models.base import (
    BatchInferenceExecutor,
    BatchModelOutput,
    ModelTranscription,
    PreparedAsrAudio,
    cuda_device,
    load_time_seconds,
)
from app.shared.asr import (
    WHISPER_IDENTIFIER,
    WHISPER_REVISION,
    AsrModelId,
    LanguageEstimate,
    TimestampedWord,
)


class WhisperxAsrModel:
    model_id = AsrModelId.WHISPERX
    package_names = ("torch", "faster-whisper")

    def __init__(self, inference_executor: BatchInferenceExecutor) -> None:
        self.inference_executor = inference_executor
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.device = cuda_device()
        model_path = snapshot_download(
            WHISPER_IDENTIFIER,
            revision=WHISPER_REVISION,
        )
        self.model = WhisperModel(model_path, device=self.device, compute_type="float16")

    def transcribe(self, audio: PreparedAsrAudio) -> ModelTranscription:
        result = self.inference_executor.execute(
            (audio.path,),
            self,
        )
        return ModelTranscription(
            words=result.outputs[0].words,
            language_estimate=result.outputs[0].language_estimate,
            inference_queue_time_seconds=result.queue_time_seconds,
            audio_loading_time_seconds=audio.loading_time_seconds,
            model_execution_time_seconds=result.execution_time_seconds,
        )

    def transcribe_batch(
        self,
        audio_paths: tuple[Path, ...],
    ) -> tuple[BatchModelOutput, ...]:
        transcriptions: list[BatchModelOutput] = []
        for audio_path in audio_paths:
            segments, transcription_info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=True,
                vad_filter=False,
            )
            words = tuple(
                TimestampedWord(
                    text=word.word,
                    start_seconds=word.start,
                    end_seconds=word.end,
                    confidence=word.probability,
                )
                for segment in segments
                for word in (segment.words or ())
            )
            transcriptions.append(
                BatchModelOutput(
                    words=words,
                    language_estimate=LanguageEstimate(
                        language_code=transcription_info.language,
                        confidence=transcription_info.language_probability,
                    ),
                )
            )
        return tuple(transcriptions)
