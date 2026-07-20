from __future__ import annotations

from transformers import AutoModelForTDT, AutoProcessor

from app.compute.asr.chunking import (
    PARAKEET_INFERENCE_BATCH_SIZE,
    ParakeetAudioChunk,
    global_chunk_words,
    parakeet_audio_chunks,
)
from app.compute.asr.models.base import (
    BatchInferenceExecutor,
    BatchModelOutput,
    ModelTranscription,
    PreparedAsrAudio,
    cuda_device,
    inference_batches,
    load_time_seconds,
)
from app.compute.asr.models.parsing import (
    words_from_parakeet_timestamps,
)
from app.shared.asr import PARAKEET_IDENTIFIER, PARAKEET_REVISION, AsrModelId, TimestampedWord


class ParakeetAsrModel:
    model_id = AsrModelId.PARAKEET_TDT
    package_names = ("torch", "transformers", "librosa")

    def __init__(self, inference_executor: BatchInferenceExecutor) -> None:
        self.inference_executor = inference_executor
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            PARAKEET_IDENTIFIER,
            revision=PARAKEET_REVISION,
        )
        self.model = AutoModelForTDT.from_pretrained(
            PARAKEET_IDENTIFIER,
            revision=PARAKEET_REVISION,
            dtype="auto",
        )
        self.device = cuda_device()
        self.model.to(self.device)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def transcribe(self, audio: PreparedAsrAudio) -> ModelTranscription:
        if audio.sample_rate != self.sample_rate:
            raise ValueError("Parakeet audio sample rate does not match the model.")
        chunks = parakeet_audio_chunks(audio.samples, audio.sample_rate)
        words: list[TimestampedWord] = []
        inference_queue_time_seconds = 0.0
        model_execution_time_seconds = 0.0
        for batch in inference_batches(chunks, PARAKEET_INFERENCE_BATCH_SIZE):
            result = self.inference_executor.execute(batch, self)
            inference_queue_time_seconds += result.queue_time_seconds
            model_execution_time_seconds += result.execution_time_seconds
            words.extend(word for output in result.outputs for word in output.words)
        return ModelTranscription(
            words=tuple(words),
            language_estimate=None,
            inference_queue_time_seconds=inference_queue_time_seconds,
            audio_loading_time_seconds=audio.loading_time_seconds,
            model_execution_time_seconds=model_execution_time_seconds,
        )

    def transcribe_batch(
        self,
        chunks: tuple[ParakeetAudioChunk, ...],
    ) -> tuple[BatchModelOutput, ...]:
        inputs = self.processor(
            [chunk.samples for chunk in chunks],
            sampling_rate=self.sample_rate,
        )
        inputs.to(self.model.device, dtype=self.model.dtype)
        output = self.model.generate(**inputs, return_dict_in_generate=True)
        _decoded_output, decoded_timestamps = self.processor.decode(
            output.sequences,
            durations=output.durations,
            skip_special_tokens=True,
        )
        if not isinstance(decoded_timestamps, list) or len(decoded_timestamps) != len(chunks):
            raise ValueError("Parakeet did not return one timestamp sequence per audio chunk.")
        return tuple(
            BatchModelOutput(
                words=global_chunk_words(
                    chunk=chunk,
                    words=tuple(words_from_parakeet_timestamps(chunk_timestamps)),
                ),
                language_estimate=None,
            )
            for chunk, chunk_timestamps in zip(chunks, decoded_timestamps, strict=True)
        )
