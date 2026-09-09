from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, TextIO, cast
from uuid import uuid4

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingForRNNT,
)
from transformers.models.nemotron_asr_streaming.processing_nemotron_asr_streaming import (
    NemotronAsrStreamingProcessor,
)

from app.compute.voice.asr_text_streamer import CumulativeAsrTextIteratorStreamer
from app.compute.voice.asr_worker_protocol import (
    AsrAudioCommand,
    AsrWorkerCommandType,
    AsrWorkerErrorEvent,
    AsrWorkerReadyEvent,
    FinalAsrEvent,
    PartialAsrEvent,
    TurnAdapterDegradedEvent,
    TurnAdapterPredictionEvent,
    asr_worker_command_adapter,
)
from app.compute.voice.schemas import (
    ActivityHorizon,
    CausalSource,
    InteractionPrediction,
    TraceStamp,
)
from app.compute.voice.turn_adapter import (
    StreamingTurnAdapter,
    load_streaming_turn_adapter,
)
from app.shared.model_constants import NEMOTRON_ASR_MODEL_NAME, NEMOTRON_ASR_MODEL_REVISION

INPUT_SAMPLE_RATE: Final = 16_000
PCM_BYTES_PER_SAMPLE: Final = 2
NUM_LOOKAHEAD_TOKENS: Final = int(os.environ.get("VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS", "1"))
TURN_ADAPTER_CHECKPOINT_ENVIRONMENT_VARIABLE: Final = "VOICE_LIGHT_TURN_ADAPTER_CHECKPOINT"


@dataclass(frozen=True)
class AudioRead:
    samples: np.ndarray
    reached_end: bool
    observation: AsrAudioCommand | None


@dataclass
class BufferedAudio:
    pcm_bytes: bytearray
    observation: AsrAudioCommand


class StreamingAudioBuffer:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.audio: deque[BufferedAudio] = deque()
        self.byte_count = 0
        self.closed = False

    def append(self, command: AsrAudioCommand) -> None:
        with self.condition:
            pcm_bytes = command.pcm_bytes()
            self.audio.append(BufferedAudio(bytearray(pcm_bytes), command))
            self.byte_count += len(pcm_bytes)
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def read(self, sample_count: int) -> AudioRead:
        byte_count = sample_count * PCM_BYTES_PER_SAMPLE
        with self.condition:
            self.condition.wait_for(lambda: self.byte_count >= byte_count or self.closed)
            consumed = bytearray()
            observation: AsrAudioCommand | None = None
            while self.audio and len(consumed) < byte_count:
                buffered = self.audio[0]
                consumed_byte_count = min(byte_count - len(consumed), len(buffered.pcm_bytes))
                consumed.extend(buffered.pcm_bytes[:consumed_byte_count])
                del buffered.pcm_bytes[:consumed_byte_count]
                self.byte_count -= consumed_byte_count
                observation = buffered.observation
                if not buffered.pcm_bytes:
                    self.audio.popleft()
            pcm_bytes = bytes(consumed)
            reached_end = self.closed and not self.audio
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32_768.0
        return AudioRead(samples=samples, reached_end=reached_end, observation=observation)


class SharedEncoderTurnInference:
    def __init__(
        self,
        model: NemotronAsrStreamingForRNNT,
        adapter: StreamingTurnAdapter,
        send_event: Callable[[TurnAdapterPredictionEvent | TurnAdapterDegradedEvent], None],
    ) -> None:
        self.adapter = adapter
        self.send_event = send_event
        self.observation: AsrAudioCommand | None = None
        self.taps: dict[int, torch.Tensor] = {}
        self.encoder_frame_count = 0
        self.active = True
        self.handles = tuple(
            model.encoder.layers[layer_index - 1].register_forward_hook(
                self._hook_for_layer(layer_index)
            )
            for layer_index in adapter.tap_layer_indices
        )
        self.adapter.reset()

    def set_observation(self, observation: AsrAudioCommand) -> None:
        self.observation = observation

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.adapter.reset()

    def _hook_for_layer(
        self,
        layer_index: int,
    ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def capture(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, inputs
            self.taps[layer_index] = output.detach()
            if not self.active or layer_index != self.adapter.tap_layer_indices[-1]:
                return
            try:
                self._emit_prediction()
            except Exception as error:
                self.active = False
                self.send_event(TurnAdapterDegradedEvent(reason=str(error)))

        return capture

    def _emit_prediction(self) -> None:
        observation = self.observation
        if observation is None:
            raise AssertionError("Encoder inference is missing its causal audio observation.")
        feature_taps = tuple(self.taps[index] for index in self.adapter.tap_layer_indices)
        frame_count = feature_taps[0].shape[1]
        probabilities = self.adapter.predict(
            feature_taps,
            assistant_speaking=observation.playback_condition.assistant_audible,
        )
        frame_start = self.encoder_frame_count
        self.encoder_frame_count += frame_count
        confidence = max(
            probabilities.turn_completion,
            probabilities.non_floor_feedback,
            probabilities.floor_take,
        )
        prediction = InteractionPrediction(
            stamp=TraceStamp(
                event_id=str(uuid4()),
                parent_event_ids=(),
                stream_epoch=observation.stream_epoch,
                turn_epoch=observation.turn_epoch,
                inference_step=observation.sequence_number,
                observation_id=f"audio:{observation.stream_epoch}:{observation.sequence_number}",
                observation_monotonic_time_ns=observation.observation_monotonic_time_ns,
                emission_monotonic_time_ns=time.perf_counter_ns(),
                encoder_frame_start=frame_start,
                encoder_frame_end=self.encoder_frame_count,
                input_start_sample=observation.start_input_sample,
                input_end_sample=observation.end_input_sample,
                observed_through_input_sample=observation.end_input_sample,
                input_sample_position=observation.end_input_sample,
                output_sample_position=(
                    observation.playback_condition.latest_output_sample_position
                ),
                conditioned_transcript_revision_id=None,
                conditioned_playback_event_id=observation.playback_condition.event_id,
                source=CausalSource.TURN_ADAPTER,
                model_name="voice-light-streaming-turn-adapter",
                model_revision=self.adapter.checkpoint_sha256,
            ),
            p_user_speech=max(probabilities.future_activity),
            p_user_yield=probabilities.user_yield,
            p_user_backchannel=probabilities.non_floor_feedback,
            p_user_interruption=probabilities.floor_take,
            p_turn_completion=probabilities.turn_completion,
            p_continuation_pause=probabilities.continuation_pause,
            future_user_activity_horizons=(
                ActivityHorizon(
                    start_ms=0, end_ms=200, probability=probabilities.future_activity[0]
                ),
                ActivityHorizon(
                    start_ms=200, end_ms=500, probability=probabilities.future_activity[1]
                ),
                ActivityHorizon(
                    start_ms=500, end_ms=1_000, probability=probabilities.future_activity[2]
                ),
                ActivityHorizon(
                    start_ms=1_000, end_ms=1_500, probability=probabilities.future_activity[3]
                ),
            ),
            assistant_playback_state=observation.playback_condition.state,
            confidence=confidence,
        )
        self.send_event(TurnAdapterPredictionEvent(prediction=prediction))


class NemotronRecognition:
    def __init__(
        self,
        processor: NemotronAsrStreamingProcessor,
        model: NemotronAsrStreamingForRNNT,
        audio_buffer: StreamingAudioBuffer,
        output_stream: TextIO,
        output_lock: threading.Lock,
        turn_adapter: StreamingTurnAdapter | None,
    ) -> None:
        self.processor = processor
        self.model = model
        self.audio_buffer = audio_buffer
        self.output_stream = output_stream
        self.output_lock = output_lock
        self.turn_adapter = turn_adapter
        self.shared_inference: SharedEncoderTurnInference | None = None

    def run(self) -> None:
        shared_inference = (
            None
            if self.turn_adapter is None
            else SharedEncoderTurnInference(self.model, self.turn_adapter, self._send_event)
        )
        self.shared_inference = shared_inference
        try:
            first_audio = self.audio_buffer.read(self.processor.num_samples_first_audio_chunk)
            if not first_audio.samples.size:
                self._send_event(FinalAsrEvent(text=""))
                return
            padded_first_audio = _pad_samples(
                first_audio.samples,
                self.processor.num_samples_first_audio_chunk,
            )
            first_inputs = self.processor(
                padded_first_audio,
                sampling_rate=INPUT_SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)
            assert first_audio.observation is not None
            if shared_inference is not None:
                shared_inference.set_observation(first_audio.observation)
            input_features = self._input_features(
                first_audio=padded_first_audio,
                first_input_features=first_inputs.input_features,
                first_audio_reached_end=first_audio.reached_end,
            )
            streamer = CumulativeAsrTextIteratorStreamer(self.processor.tokenizer)
            generation_errors: list[Exception] = []
            generate = partial(
                self.model.generate,
                input_features=input_features,
                attention_mask=first_inputs.attention_mask,
                num_lookahead_tokens=first_inputs.num_lookahead_tokens,
                streamer=streamer,
            )

            def run_generation() -> None:
                try:
                    generate()
                except Exception as error:
                    generation_errors.append(error)
                    streamer.on_finalized_text("", stream_end=True)

            generation_thread = threading.Thread(target=run_generation, daemon=True)
            generation_thread.start()
            transcript = ""
            for cumulative_text in streamer:
                normalized_text = cumulative_text.strip()
                if not normalized_text or normalized_text == transcript:
                    continue
                transcript = normalized_text
                self._send_event(PartialAsrEvent(text=transcript))
            generation_thread.join()
            if generation_errors:
                raise generation_errors[0]
            self._send_event(FinalAsrEvent(text=transcript))
        except Exception as error:
            self._send_event(AsrWorkerErrorEvent(message=str(error)))
        finally:
            if shared_inference is not None:
                shared_inference.close()
            self.shared_inference = None

    def _input_features(
        self,
        first_audio: np.ndarray,
        first_input_features: torch.Tensor,
        first_audio_reached_end: bool,
    ) -> Iterator[torch.Tensor]:
        yield first_input_features[:, : self.processor.num_mel_frames_first_audio_chunk, :]
        if first_audio_reached_end:
            return
        overlap_sample_count = self.processor.feature_extractor.n_fft // 2
        previous_tail = first_audio[-overlap_sample_count:]
        new_sample_count = self.processor.num_samples_per_audio_chunk - overlap_sample_count
        while True:
            audio_read = self.audio_buffer.read(new_sample_count)
            if not audio_read.samples.size:
                return
            audio_chunk = np.concatenate((previous_tail, audio_read.samples))
            audio_chunk = _pad_samples(audio_chunk, self.processor.num_samples_per_audio_chunk)
            inputs = self.processor(
                audio_chunk,
                sampling_rate=INPUT_SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)
            assert audio_read.observation is not None
            if self.shared_inference is not None:
                self.shared_inference.set_observation(audio_read.observation)
            yield inputs.input_features
            previous_tail = audio_chunk[-overlap_sample_count:]
            if audio_read.reached_end:
                return

    def _send_event(
        self,
        event: (
            PartialAsrEvent
            | FinalAsrEvent
            | AsrWorkerErrorEvent
            | TurnAdapterPredictionEvent
            | TurnAdapterDegradedEvent
        ),
    ) -> None:
        with self.output_lock:
            self.output_stream.write(event.model_dump_json() + "\n")
            self.output_stream.flush()


def main() -> None:
    processor = cast(
        NemotronAsrStreamingProcessor,
        AutoProcessor.from_pretrained(
            NEMOTRON_ASR_MODEL_NAME,
            revision=NEMOTRON_ASR_MODEL_REVISION,
        ),
    )
    processor.set_num_lookahead_tokens(NUM_LOOKAHEAD_TOKENS)
    model = cast(
        NemotronAsrStreamingForRNNT,
        AutoModelForRNNT.from_pretrained(
            NEMOTRON_ASR_MODEL_NAME,
            revision=NEMOTRON_ASR_MODEL_REVISION,
            dtype=torch.bfloat16,
        ).to("cuda"),
    )
    turn_adapter, turn_adapter_error = _load_turn_adapter(model)
    output_lock = threading.Lock()
    with output_lock:
        sys.stdout.write(
            AsrWorkerReadyEvent(
                first_prediction_audio_samples=processor.num_samples_first_audio_chunk,
                subsequent_prediction_audio_samples=(
                    processor.num_samples_per_audio_chunk - processor.feature_extractor.n_fft // 2
                ),
                turn_adapter_available=turn_adapter is not None,
                turn_adapter_checkpoint_sha256=(
                    None if turn_adapter is None else turn_adapter.checkpoint_sha256
                ),
                turn_adapter_error=turn_adapter_error,
            ).model_dump_json()
            + "\n"
        )
        sys.stdout.flush()
    audio_buffer: StreamingAudioBuffer | None = None
    recognition_thread: threading.Thread | None = None

    for command_json in sys.stdin:
        command = asr_worker_command_adapter.validate_json(command_json)
        match command.type:
            case AsrWorkerCommandType.START:
                if recognition_thread is not None and recognition_thread.is_alive():
                    _send_controller_error("An ASR session is already active.", output_lock)
                    continue
                audio_buffer = StreamingAudioBuffer()
                recognition = NemotronRecognition(
                    processor=processor,
                    model=model,
                    audio_buffer=audio_buffer,
                    output_stream=sys.stdout,
                    output_lock=output_lock,
                    turn_adapter=turn_adapter,
                )
                recognition_thread = threading.Thread(target=recognition.run, daemon=True)
                recognition_thread.start()
            case AsrWorkerCommandType.AUDIO:
                assert isinstance(command, AsrAudioCommand)
                if audio_buffer is None:
                    _send_controller_error("No ASR session is active.", output_lock)
                    continue
                audio_buffer.append(command)
            case AsrWorkerCommandType.FINISH:
                if audio_buffer is None or recognition_thread is None:
                    _send_controller_error("No ASR session is active.", output_lock)
                    continue
                audio_buffer.close()
                recognition_thread.join()
                audio_buffer = None
                recognition_thread = None
            case AsrWorkerCommandType.SHUTDOWN:
                if audio_buffer is not None:
                    audio_buffer.close()
                if recognition_thread is not None:
                    recognition_thread.join()
                return


def _pad_samples(samples: np.ndarray, sample_count: int) -> np.ndarray:
    if len(samples) >= sample_count:
        return samples
    return np.pad(samples, (0, sample_count - len(samples)))


def _send_controller_error(message: str, output_lock: threading.Lock) -> None:
    with output_lock:
        sys.stdout.write(AsrWorkerErrorEvent(message=message).model_dump_json() + "\n")
        sys.stdout.flush()


def _load_turn_adapter(
    model: NemotronAsrStreamingForRNNT,
) -> tuple[StreamingTurnAdapter | None, str | None]:
    checkpoint_value = os.environ.get(TURN_ADAPTER_CHECKPOINT_ENVIRONMENT_VARIABLE)
    if checkpoint_value is None:
        return None, None
    try:
        loaded = load_streaming_turn_adapter(
            checkpoint_path=Path(checkpoint_value),
            expected_model_identifier=NEMOTRON_ASR_MODEL_NAME,
            expected_model_revision=NEMOTRON_ASR_MODEL_REVISION,
            expected_lookahead_tokens=NUM_LOOKAHEAD_TOKENS,
            device=model.device,
        )
    except Exception as error:
        return None, str(error)
    adapter = StreamingTurnAdapter(loaded)
    adapter.warm_up()
    return adapter, None


if __name__ == "__main__":
    main()
