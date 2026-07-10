from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from transformers import WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature

from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.audio.wav import read_mono_wave_audio

MODEL_REPOSITORY = "pipecat-ai/smart-turn-v3"
MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
MODEL_SAMPLE_RATE = 16000
MAX_MODEL_WINDOW_SECONDS = 8.0


class SmartTurnFeatureExtractor(Protocol):
    def __call__(
        self,
        raw_speech: NDArray[np.float32],
        sampling_rate: int,
        return_tensors: str,
        padding: str,
        max_length: int,
        truncation: bool,
        do_normalize: bool,
    ) -> BatchFeature:
        raise NotImplementedError


class SmartTurnSession(Protocol):
    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        raise NotImplementedError


@dataclass(frozen=True)
class SmartTurnV3Inference:
    feature_extractor: SmartTurnFeatureExtractor
    session: SmartTurnSession


@dataclass(frozen=True)
class CandidatePause:
    speech_start_seconds: float
    speech_end_seconds: float
    evaluate_seconds: float
    following_silence_seconds: float


@dataclass(frozen=True)
class PipecatSmartTurnV3Detector:
    info: EndOfTurnDetectorInfo
    frame_seconds: float
    min_speech_seconds: float
    candidate_silence_seconds: float
    completion_threshold: float

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        return run_pipecat_smart_turn_v3(
            wave_path=speaker1_path,
            result_name=self.info.mode.value,
            description=self.info.description,
            frame_seconds=self.frame_seconds,
            min_speech_seconds=self.min_speech_seconds,
            candidate_silence_seconds=self.candidate_silence_seconds,
            completion_threshold=self.completion_threshold,
        )


def pipecat_smart_turn_v3_detector() -> PipecatSmartTurnV3Detector:
    return PipecatSmartTurnV3Detector(
        info=EndOfTurnDetectorInfo(
            mode=EndOfTurnDetectorMode.PIPECAT_SMART_TURN_V3,
            label="Pipecat Smart Turn v3.2",
            description=(
                "RMS speech windows propose candidate turn boundaries after 200 ms of silence; "
                "Pipecat Smart Turn v3.2 CPU ONNX then scores the latest 8 s of 16 kHz mono "
                "audio and emits an end-of-turn event when completion probability is >= 0.5."
            ),
        ),
        frame_seconds=0.03,
        min_speech_seconds=0.09,
        candidate_silence_seconds=0.2,
        completion_threshold=0.5,
    )


def run_pipecat_smart_turn_v3(
    wave_path: Path,
    result_name: str,
    description: str,
    frame_seconds: float = 0.03,
    min_speech_seconds: float = 0.09,
    candidate_silence_seconds: float = 0.2,
    completion_threshold: float = 0.5,
) -> BaselineResult:
    audio = _read_normalized_audio(
        wave_path=wave_path,
        target_sample_rate=MODEL_SAMPLE_RATE,
    )
    frame_energies = _frame_energies(
        samples=audio.samples,
        sample_rate=audio.sample_rate,
        frame_seconds=frame_seconds,
    )
    threshold = _adaptive_threshold(frame_energies=frame_energies)
    speech_flags = [frame_energy >= threshold for frame_energy in frame_energies]
    speech_segments = _speech_segments_from_flags_with_pause(
        speech_flags=speech_flags,
        frame_seconds=frame_seconds,
        min_speech_seconds=min_speech_seconds,
        candidate_silence_seconds=candidate_silence_seconds,
    )
    inference = _load_smart_turn_v3_inference(
        model_repository=MODEL_REPOSITORY,
        model_filename=MODEL_FILENAME,
    )
    candidate_pauses = _candidate_pauses(
        speech_segments=speech_segments,
        audio_duration_seconds=audio.duration_seconds,
        candidate_silence_seconds=candidate_silence_seconds,
    )
    end_of_turn_events = _smart_turn_events(
        inference=inference,
        samples=audio.samples,
        sample_rate=audio.sample_rate,
        candidate_pauses=candidate_pauses,
        completion_threshold=completion_threshold,
    )
    return BaselineResult(
        name=result_name,
        description=description,
        frame_seconds=frame_seconds,
        min_silence_seconds=candidate_silence_seconds,
        threshold=completion_threshold,
        speech_segments=speech_segments,
        pause_spans=[],
        backchannel_spans=[],
        end_of_turn_events=end_of_turn_events,
    )


@dataclass(frozen=True)
class NormalizedAudio:
    samples: NDArray[np.float32]
    sample_rate: int
    duration_seconds: float


def _read_normalized_audio(
    wave_path: Path,
    target_sample_rate: int,
) -> NormalizedAudio:
    audio = read_mono_wave_audio(wave_path=wave_path)
    maximum_amplitude = float((1 << (audio.sample_width * 8 - 1)) - 1)
    if audio.samples.size == 0:
        raise ValueError("Cannot run Pipecat Smart Turn v3 on an audio file with no frames.")
    normalized_samples = np.clip(audio.samples / maximum_amplitude, -1.0, 1.0).astype(np.float32)
    resampled_samples = _resample_audio(
        samples=normalized_samples,
        source_sample_rate=audio.sample_rate,
        target_sample_rate=target_sample_rate,
    )
    return NormalizedAudio(
        samples=resampled_samples,
        sample_rate=target_sample_rate,
        duration_seconds=resampled_samples.size / target_sample_rate,
    )


def _resample_audio(
    samples: NDArray[np.float32],
    source_sample_rate: int,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    if source_sample_rate == target_sample_rate:
        return samples

    duration_seconds = samples.size / source_sample_rate
    target_sample_count = max(1, round(duration_seconds * target_sample_rate))
    source_positions = np.linspace(0.0, duration_seconds, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, duration_seconds, num=target_sample_count, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _frame_energies(
    samples: NDArray[np.float32],
    sample_rate: int,
    frame_seconds: float,
) -> list[float]:
    samples_per_frame = max(1, round(sample_rate * frame_seconds))
    frame_energies: list[float] = []
    for start_index in range(0, samples.size, samples_per_frame):
        frame = samples[start_index : start_index + samples_per_frame]
        frame_energies.append(float(np.sqrt(np.mean(frame * frame))))
    return frame_energies


def _adaptive_threshold(frame_energies: list[float]) -> float:
    if not frame_energies:
        raise ValueError("Cannot calculate a speech threshold for an empty audio file.")

    sorted_energies = sorted(frame_energies)
    noise_floor = _percentile(sorted_values=sorted_energies, percentile=0.20)
    speech_level = _percentile(sorted_values=sorted_energies, percentile=0.90)
    return max(noise_floor * 3.0, speech_level * 0.08, 0.00003)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile for an empty list.")
    bounded_percentile = min(1.0, max(0.0, percentile))
    index = round((len(sorted_values) - 1) * bounded_percentile)
    return sorted_values[index]


def _speech_segments_from_flags_with_pause(
    speech_flags: list[bool],
    frame_seconds: float,
    min_speech_seconds: float,
    candidate_silence_seconds: float,
) -> list[SpeechSegment]:
    speech_segments: list[SpeechSegment] = []
    active_start_index: int | None = None
    last_speech_index: int | None = None
    silence_frame_count = 0
    required_silence_frames = max(1, round(candidate_silence_seconds / frame_seconds))

    for frame_index, is_speech in enumerate(speech_flags):
        if is_speech:
            if active_start_index is None:
                active_start_index = frame_index
            last_speech_index = frame_index
            silence_frame_count = 0
            continue

        if active_start_index is not None:
            silence_frame_count += 1
            if silence_frame_count >= required_silence_frames:
                assert last_speech_index is not None
                _append_speech_segment(
                    speech_segments=speech_segments,
                    start_index=active_start_index,
                    end_index=last_speech_index + 1,
                    frame_seconds=frame_seconds,
                    min_speech_seconds=min_speech_seconds,
                )
                active_start_index = None
                last_speech_index = None
                silence_frame_count = 0

    if active_start_index is not None:
        assert last_speech_index is not None
        _append_speech_segment(
            speech_segments=speech_segments,
            start_index=active_start_index,
            end_index=last_speech_index + 1,
            frame_seconds=frame_seconds,
            min_speech_seconds=min_speech_seconds,
        )

    return speech_segments


def _append_speech_segment(
    speech_segments: list[SpeechSegment],
    start_index: int,
    end_index: int,
    frame_seconds: float,
    min_speech_seconds: float,
) -> None:
    start_seconds = _rounded_seconds(start_index * frame_seconds)
    end_seconds = _rounded_seconds(end_index * frame_seconds)
    if _rounded_seconds(end_seconds - start_seconds) >= min_speech_seconds:
        speech_segments.append(SpeechSegment(start_seconds=start_seconds, end_seconds=end_seconds))


def _candidate_pauses(
    speech_segments: list[SpeechSegment],
    audio_duration_seconds: float,
    candidate_silence_seconds: float,
) -> list[CandidatePause]:
    candidate_pauses: list[CandidatePause] = []

    for current_segment, next_segment in zip(
        speech_segments,
        speech_segments[1:],
        strict=False,
    ):
        following_silence_seconds = next_segment.start_seconds - current_segment.end_seconds
        if following_silence_seconds >= candidate_silence_seconds:
            candidate_pauses.append(
                CandidatePause(
                    speech_start_seconds=current_segment.start_seconds,
                    speech_end_seconds=current_segment.end_seconds,
                    evaluate_seconds=_rounded_seconds(
                        current_segment.end_seconds + candidate_silence_seconds
                    ),
                    following_silence_seconds=_rounded_seconds(following_silence_seconds),
                )
            )

    if speech_segments:
        final_segment = speech_segments[-1]
        trailing_silence_seconds = audio_duration_seconds - final_segment.end_seconds
        if trailing_silence_seconds >= candidate_silence_seconds:
            candidate_pauses.append(
                CandidatePause(
                    speech_start_seconds=final_segment.start_seconds,
                    speech_end_seconds=final_segment.end_seconds,
                    evaluate_seconds=_rounded_seconds(
                        final_segment.end_seconds + candidate_silence_seconds
                    ),
                    following_silence_seconds=_rounded_seconds(trailing_silence_seconds),
                )
            )

    return candidate_pauses


def _smart_turn_events(
    inference: SmartTurnV3Inference,
    samples: NDArray[np.float32],
    sample_rate: int,
    candidate_pauses: list[CandidatePause],
    completion_threshold: float,
) -> list[EndOfTurnEvent]:
    events: list[EndOfTurnEvent] = []
    turn_start_seconds: float | None = None
    for candidate_pause in candidate_pauses:
        if turn_start_seconds is None:
            turn_start_seconds = candidate_pause.speech_start_seconds
        start_sample = _seconds_to_sample_index(
            seconds=turn_start_seconds,
            sample_rate=sample_rate,
        )
        end_sample = _seconds_to_sample_index(
            seconds=candidate_pause.evaluate_seconds,
            sample_rate=sample_rate,
        )
        completion_probability = _completion_probability(
            inference=inference,
            audio=samples[start_sample:end_sample],
        )
        if completion_probability >= completion_threshold:
            events.append(
                EndOfTurnEvent(
                    time_seconds=candidate_pause.evaluate_seconds,
                    speech_start_seconds=candidate_pause.speech_start_seconds,
                    speech_end_seconds=candidate_pause.speech_end_seconds,
                    silence_seconds=candidate_pause.following_silence_seconds,
                )
            )
            turn_start_seconds = None
    return events


def _seconds_to_sample_index(seconds: float, sample_rate: int) -> int:
    return max(0, round(seconds * sample_rate))


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, 6)


@lru_cache(maxsize=1)
def _load_smart_turn_v3_inference(
    model_repository: str,
    model_filename: str,
) -> SmartTurnV3Inference:
    model_path = hf_hub_download(
        repo_id=model_repository,
        filename=model_filename,
    )
    session_options = ort.SessionOptions()
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = cast(SmartTurnSession, ort.InferenceSession(model_path, sess_options=session_options))
    feature_extractor = cast(SmartTurnFeatureExtractor, WhisperFeatureExtractor(chunk_length=8))
    return SmartTurnV3Inference(feature_extractor=feature_extractor, session=session)


def _completion_probability(
    inference: SmartTurnV3Inference,
    audio: NDArray[np.float32],
) -> float:
    model_audio = _padded_model_audio(audio=audio)
    feature_batch = inference.feature_extractor(
        model_audio,
        sampling_rate=MODEL_SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=round(MAX_MODEL_WINDOW_SECONDS * MODEL_SAMPLE_RATE),
        truncation=True,
        do_normalize=True,
    )
    input_features = cast(NDArray[np.float32], feature_batch["input_features"])
    squeezed_features = input_features.squeeze(0).astype(np.float32)
    batched_features = np.expand_dims(squeezed_features, axis=0)
    outputs = inference.session.run(None, {"input_features": batched_features})
    probabilities = cast(NDArray[np.float32], outputs[0])
    return float(probabilities[0].item())


def _padded_model_audio(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    max_sample_count = round(MAX_MODEL_WINDOW_SECONDS * MODEL_SAMPLE_RATE)
    if audio.size > max_sample_count:
        return audio[-max_sample_count:]
    if audio.size < max_sample_count:
        padding_sample_count = max_sample_count - audio.size
        return np.pad(audio, (padding_sample_count, 0), mode="constant", constant_values=0.0)
    return audio
