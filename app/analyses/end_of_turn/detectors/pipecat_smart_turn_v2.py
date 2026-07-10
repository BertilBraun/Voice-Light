from __future__ import annotations

import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.audio.wav import mono_samples

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.modeling_outputs import SequenceClassifierOutput


MODEL_NAME = "pipecat-ai/smart-turn-v2"
TARGET_SAMPLE_RATE = 16000
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
    ) -> BatchFeature:
        raise NotImplementedError


class SmartTurnModel(Protocol):
    def eval(self) -> Module:
        raise NotImplementedError

    def __call__(self, input_values: Tensor) -> SequenceClassifierOutput:
        raise NotImplementedError


@dataclass(frozen=True)
class SmartTurnInferenceComponents:
    feature_extractor: SmartTurnFeatureExtractor
    model: SmartTurnModel


@dataclass(frozen=True)
class PipecatSmartTurnV2Detector:
    info: EndOfTurnDetectorInfo
    model_name: str
    completion_threshold: float
    frame_seconds: float
    min_speech_seconds: float
    min_silence_seconds: float
    max_window_seconds: float

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        inference_components = _load_inference_components(model_name=self.model_name)
        audio = _read_mono_float_audio(
            wave_path=speaker1_path,
            target_sample_rate=TARGET_SAMPLE_RATE,
        )
        speech_segments = _candidate_speech_segments(
            audio=audio,
            sample_rate=TARGET_SAMPLE_RATE,
            frame_seconds=self.frame_seconds,
            min_speech_seconds=self.min_speech_seconds,
            merge_gap_seconds=self.min_silence_seconds,
        )
        end_of_turn_events = _smart_turn_events(
            audio=audio,
            sample_rate=TARGET_SAMPLE_RATE,
            speech_segments=speech_segments,
            inference_components=inference_components,
            completion_threshold=self.completion_threshold,
            min_silence_seconds=self.min_silence_seconds,
            max_window_seconds=self.max_window_seconds,
        )
        return BaselineResult(
            name=self.info.mode.value,
            description=self.info.description,
            frame_seconds=self.frame_seconds,
            min_silence_seconds=self.min_silence_seconds,
            threshold=self.completion_threshold,
            speech_segments=speech_segments,
            end_of_turn_events=end_of_turn_events,
        )


def pipecat_smart_turn_v2_detector(
    mode: EndOfTurnDetectorMode,
) -> PipecatSmartTurnV2Detector:
    return PipecatSmartTurnV2Detector(
        info=EndOfTurnDetectorInfo(
            mode=mode,
            label="Pipecat Smart Turn v2",
            description=(
                "RMS speech windows propose candidate turn boundaries after 400 ms of silence; "
                "Pipecat Smart Turn v2 then runs on each candidate's latest 8 s of 16 kHz mono "
                "audio and emits an end-of-turn event only when completion probability is >= 0.5."
            ),
        ),
        model_name=MODEL_NAME,
        completion_threshold=0.5,
        frame_seconds=0.03,
        min_speech_seconds=0.09,
        min_silence_seconds=0.4,
        max_window_seconds=MAX_MODEL_WINDOW_SECONDS,
    )


@lru_cache(maxsize=1)
def _load_inference_components(model_name: str) -> SmartTurnInferenceComponents:
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    except ImportError as error:
        raise ImportError(
            "Pipecat Smart Turn v2 requires optional dependencies: "
            "`torch`, `transformers`, and `safetensors`. Install them before enabling "
            "this detector."
        ) from error

    try:
        feature_extractor = cast(
            SmartTurnFeatureExtractor,
            Wav2Vec2FeatureExtractor.from_pretrained(model_name),
        )
        model = cast(
            SmartTurnModel,
            Wav2Vec2ForSequenceClassification.from_pretrained(model_name),
        )
    except OSError as error:
        raise ValueError(
            f"Could not load Pipecat Smart Turn v2 model `{model_name}`. "
            "Check network access, Hugging Face cache configuration, and model availability."
        ) from error

    model.eval()
    return SmartTurnInferenceComponents(feature_extractor=feature_extractor, model=model)


def _read_mono_float_audio(
    wave_path: Path,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    with wave.open(str(wave_path), "rb") as wave_reader:
        sample_rate = wave_reader.getframerate()
        sample_width = wave_reader.getsampwidth()
        channel_count = wave_reader.getnchannels()
        fragment = wave_reader.readframes(wave_reader.getnframes())

    samples = mono_samples(
        fragment=fragment,
        sample_width=sample_width,
        channel_count=channel_count,
    )
    normalized_samples = _normalize_pcm_samples(samples=samples, sample_width=sample_width)
    return _resample_audio(
        audio=normalized_samples,
        source_sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
    )


def _normalize_pcm_samples(
    samples: NDArray[np.float64],
    sample_width: int,
) -> NDArray[np.float32]:
    maximum_amplitude = float((1 << (sample_width * 8 - 1)) - 1)
    clipped_samples = np.clip(samples / maximum_amplitude, -1.0, 1.0)
    return clipped_samples.astype(np.float32)


def _resample_audio(
    audio: NDArray[np.float32],
    source_sample_rate: int,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    if source_sample_rate == target_sample_rate:
        return audio
    if len(audio) == 0:
        raise ValueError("Cannot run Pipecat Smart Turn v2 on an audio file with no frames.")

    duration_seconds = len(audio) / source_sample_rate
    target_sample_count = max(1, round(duration_seconds * target_sample_rate))
    source_positions = np.linspace(0.0, duration_seconds, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, duration_seconds, num=target_sample_count, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _candidate_speech_segments(
    audio: NDArray[np.float32],
    sample_rate: int,
    frame_seconds: float,
    min_speech_seconds: float,
    merge_gap_seconds: float,
) -> list[SpeechSegment]:
    frame_energies = _frame_energies(
        audio=audio,
        sample_rate=sample_rate,
        frame_seconds=frame_seconds,
    )
    threshold = _adaptive_threshold(frame_energies=frame_energies)
    speech_flags = [frame_energy >= threshold for frame_energy in frame_energies]
    raw_segments = _speech_segments_from_flags(
        speech_flags=speech_flags,
        frame_seconds=frame_seconds,
        min_speech_seconds=min_speech_seconds,
    )
    return _merge_close_segments(
        speech_segments=raw_segments,
        merge_gap_seconds=merge_gap_seconds,
    )


def _frame_energies(
    audio: NDArray[np.float32],
    sample_rate: int,
    frame_seconds: float,
) -> list[float]:
    if len(audio) == 0:
        raise ValueError("Cannot run Pipecat Smart Turn v2 on an audio file with no frames.")

    frames_per_window = max(1, round(sample_rate * frame_seconds))
    frame_energies: list[float] = []
    for start_index in range(0, len(audio), frames_per_window):
        frame = audio[start_index : start_index + frames_per_window]
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


def _speech_segments_from_flags(
    speech_flags: list[bool],
    frame_seconds: float,
    min_speech_seconds: float,
) -> list[SpeechSegment]:
    speech_segments: list[SpeechSegment] = []
    active_start_index: int | None = None

    for frame_index, is_speech in enumerate(speech_flags):
        if is_speech and active_start_index is None:
            active_start_index = frame_index
        if not is_speech and active_start_index is not None:
            _append_segment_if_long_enough(
                speech_segments=speech_segments,
                start_index=active_start_index,
                end_index=frame_index,
                frame_seconds=frame_seconds,
                min_speech_seconds=min_speech_seconds,
            )
            active_start_index = None

    if active_start_index is not None:
        _append_segment_if_long_enough(
            speech_segments=speech_segments,
            start_index=active_start_index,
            end_index=len(speech_flags),
            frame_seconds=frame_seconds,
            min_speech_seconds=min_speech_seconds,
        )

    return speech_segments


def _append_segment_if_long_enough(
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


def _merge_close_segments(
    speech_segments: list[SpeechSegment],
    merge_gap_seconds: float,
) -> list[SpeechSegment]:
    if not speech_segments:
        return []

    merged_segments = [speech_segments[0]]
    for speech_segment in speech_segments[1:]:
        previous_segment = merged_segments[-1]
        gap_seconds = speech_segment.start_seconds - previous_segment.end_seconds
        if gap_seconds <= merge_gap_seconds:
            merged_segments[-1] = SpeechSegment(
                start_seconds=previous_segment.start_seconds,
                end_seconds=speech_segment.end_seconds,
            )
        else:
            merged_segments.append(speech_segment)

    return merged_segments


def _smart_turn_events(
    audio: NDArray[np.float32],
    sample_rate: int,
    speech_segments: list[SpeechSegment],
    inference_components: SmartTurnInferenceComponents,
    completion_threshold: float,
    min_silence_seconds: float,
    max_window_seconds: float,
) -> list[EndOfTurnEvent]:
    audio_duration_seconds = len(audio) / sample_rate
    end_of_turn_events: list[EndOfTurnEvent] = []

    for segment_index, speech_segment in enumerate(speech_segments):
        next_segment_start_seconds = _next_segment_start_seconds(
            speech_segments=speech_segments,
            segment_index=segment_index,
            audio_duration_seconds=audio_duration_seconds,
        )
        silence_seconds = next_segment_start_seconds - speech_segment.end_seconds
        if silence_seconds < min_silence_seconds:
            continue

        event_time_seconds = _rounded_seconds(speech_segment.end_seconds + min_silence_seconds)
        window_audio = _model_window_audio(
            audio=audio,
            sample_rate=sample_rate,
            end_seconds=event_time_seconds,
            max_window_seconds=max_window_seconds,
        )
        completion_probability = _completion_probability(
            audio=window_audio,
            inference_components=inference_components,
        )
        if completion_probability >= completion_threshold:
            end_of_turn_events.append(
                EndOfTurnEvent(
                    time_seconds=event_time_seconds,
                    speech_start_seconds=speech_segment.start_seconds,
                    speech_end_seconds=speech_segment.end_seconds,
                    silence_seconds=_rounded_seconds(silence_seconds),
                )
            )

    return end_of_turn_events


def _next_segment_start_seconds(
    speech_segments: list[SpeechSegment],
    segment_index: int,
    audio_duration_seconds: float,
) -> float:
    next_segment_index = segment_index + 1
    if next_segment_index < len(speech_segments):
        return speech_segments[next_segment_index].start_seconds
    return audio_duration_seconds


def _model_window_audio(
    audio: NDArray[np.float32],
    sample_rate: int,
    end_seconds: float,
    max_window_seconds: float,
) -> NDArray[np.float32]:
    end_index = min(len(audio), round(end_seconds * sample_rate))
    max_window_sample_count = round(max_window_seconds * sample_rate)
    start_index = max(0, end_index - max_window_sample_count)
    window_audio = audio[start_index:end_index]
    if len(window_audio) == 0:
        raise ValueError("Cannot run Pipecat Smart Turn v2 on an empty model window.")
    return window_audio


def _completion_probability(
    audio: NDArray[np.float32],
    inference_components: SmartTurnInferenceComponents,
) -> float:
    import torch

    feature_batch = inference_components.feature_extractor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=round(MAX_MODEL_WINDOW_SECONDS * TARGET_SAMPLE_RATE),
        truncation=True,
    )
    input_values = cast("Tensor", feature_batch["input_values"])
    with torch.no_grad():
        model_output = inference_components.model(input_values=input_values)
        probability = torch.sigmoid(model_output.logits).squeeze()

    return float(probability.detach().cpu().item())


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, 6)
