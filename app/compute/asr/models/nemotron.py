from __future__ import annotations

from torch import Tensor
from transformers import AutoModelForRNNT, AutoProcessor

from app.compute.asr.models.base import (
    BatchInferenceExecutor,
    ModelTranscription,
    PreparedAsrAudio,
    cuda_device,
    load_time_seconds,
)
from app.shared.asr import (
    NEMOTRON_3_5_IDENTIFIER,
    NEMOTRON_3_5_REVISION,
    AsrModelId,
    TimestampedWord,
)


class NemotronAsrModel:
    model_id = AsrModelId.NEMOTRON_3_5
    package_names = ("torch", "transformers", "librosa")

    def __init__(self, inference_executor: BatchInferenceExecutor) -> None:
        self.inference_executor = inference_executor
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            NEMOTRON_3_5_IDENTIFIER,
            revision=NEMOTRON_3_5_REVISION,
        )
        self.model = AutoModelForRNNT.from_pretrained(
            NEMOTRON_3_5_IDENTIFIER,
            revision=NEMOTRON_3_5_REVISION,
            dtype="auto",
        )
        self.device = cuda_device()
        self.model.to(self.device)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def transcribe(self, audio: PreparedAsrAudio) -> ModelTranscription:
        if audio.sample_rate != self.sample_rate:
            raise ValueError("Nemotron audio sample rate does not match the model.")
        result = self.inference_executor.execute(
            (audio,),
            self,
        )
        return ModelTranscription(
            words=result.outputs[0],
            inference_queue_time_seconds=result.queue_time_seconds,
            audio_loading_time_seconds=audio.loading_time_seconds,
            model_execution_time_seconds=result.execution_time_seconds,
        )

    def transcribe_batch(
        self,
        audio_batch: tuple[PreparedAsrAudio, ...],
    ) -> tuple[tuple[TimestampedWord, ...], ...]:
        transcriptions: list[tuple[TimestampedWord, ...]] = []
        for audio in audio_batch:
            inputs = self.processor(
                audio.samples,
                sampling_rate=self.sample_rate,
                language="en-US",
                return_tensors="pt",
            )
            inputs.to(self.model.device, dtype=self.model.dtype)
            output = self.model.generate(**inputs, return_dict_in_generate=True)
            decoded_text = self.decoded_text(output.sequences)
            transcriptions.append(
                ()
                if not decoded_text.strip()
                else (
                    TimestampedWord(
                        text=decoded_text.strip(),
                        start_seconds=0.0,
                        end_seconds=audio.duration_seconds,
                    ),
                )
            )
        return tuple(transcriptions)

    def decoded_text(self, sequences: Tensor) -> str:
        decoded = self.processor.decode(sequences, skip_special_tokens=True)
        if isinstance(decoded, str):
            return decoded
        return str(decoded[0])
