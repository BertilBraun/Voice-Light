from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
from livekit.local_inference import EOT, EOT_MAX_SAMPLES, VAD, VAD_WINDOW_SAMPLES
from numpy.typing import NDArray

from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.audio.wav import read_mono_wave_audio

MODEL_SAMPLE_RATE = 16000
VAD_SPEECH_THRESHOLD = 0.5
END_OF_TURN_THRESHOLD_EN = 0.36
CANDIDATE_SILENCE_SECONDS = 0.3
MIN_SPEECH_SECONDS = 0.09


class LiveKitEndOfTurnModel(Protocol):
    def predict(self, pcm: NDArray[np.int16]) -> float:
        raise NotImplementedError


class LiveKitVadModel(Protocol):
    def predict(self, pcm: NDArray[np.int16]) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class LiveKitV1MiniInference:
    vad_model: LiveKitVadModel
    end_of_turn_model: LiveKitEndOfTurnModel


@dataclass(frozen=True)
class NormalizedPcmAudio:
    samples: NDArray[np.int16]
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True)
class CandidatePause:
    speech_start_seconds: float
    speech_end_seconds: float
    evaluate_seconds: float
    following_silence_seconds: float


@dataclass(frozen=True)
class LiveKitV1MiniDetector:
    info: EndOfTurnDetectorInfo
    vad_speech_threshold: float
    end_of_turn_threshold: float
    min_speech_seconds: float
    candidate_silence_seconds: float

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        return run_livekit_v1_mini(
            wave_path=speaker1_path,
            result_name=self.info.mode.value,
            description=self.info.description,
            vad_speech_threshold=self.vad_speech_threshold,
            end_of_turn_threshold=self.end_of_turn_threshold,
            min_speech_seconds=self.min_speech_seconds,
            candidate_silence_seconds=self.candidate_silence_seconds,
        )


def livekit_v1_mini_detector() -> LiveKitV1MiniDetector:
    return LiveKitV1MiniDetector(
        info=EndOfTurnDetectorInfo(
            mode=EndOfTurnDetectorMode.LIVEKIT_V1_MINI,
            label="LiveKit Turn Detector v1-mini",
            description=(
                "LiveKit local-inference VAD proposes candidate turn boundaries after "
                "300 ms of silence; LiveKit Turn Detector v1-mini then scores the latest "
                "1.2 s of 16 kHz mono PCM and emits an end-of-turn event when the "
                "English calibrated probability is >= 0.36."
            ),
        ),
        vad_speech_threshold=VAD_SPEECH_THRESHOLD,
        end_of_turn_threshold=END_OF_TURN_THRESHOLD_EN,
        min_speech_seconds=MIN_SPEECH_SECONDS,
        candidate_silence_seconds=CANDIDATE_SILENCE_SECONDS,
    )


def run_livekit_v1_mini(
    wave_path: Path,
    result_name: str,
    description: str,
    vad_speech_threshold: float = VAD_SPEECH_THRESHOLD,
    end_of_turn_threshold: float = END_OF_TURN_THRESHOLD_EN,
    min_speech_seconds: float = MIN_SPEECH_SECONDS,
    candidate_silence_seconds: float = CANDIDATE_SILENCE_SECONDS,
) -> BaselineResult:
    audio = _read_16khz_int16_audio(wave_path=wave_path)
    inference = _load_livekit_v1_mini_inference()
    speech_segments = _speech_segments_from_vad(
        samples=audio.samples,
        sample_rate=audio.sample_rate,
        vad_model=inference.vad_model,
        vad_speech_threshold=vad_speech_threshold,
        min_speech_seconds=min_speech_seconds,
        candidate_silence_seconds=candidate_silence_seconds,
    )
    candidate_pauses = _candidate_pauses(
        speech_segments=speech_segments,
        audio_duration_seconds=audio.duration_seconds,
        candidate_silence_seconds=candidate_silence_seconds,
    )
    end_of_turn_events = _end_of_turn_events(
        samples=audio.samples,
        sample_rate=audio.sample_rate,
        candidate_pauses=candidate_pauses,
        end_of_turn_model=inference.end_of_turn_model,
        end_of_turn_threshold=end_of_turn_threshold,
    )
    return BaselineResult(
        name=result_name,
        description=description,
        frame_seconds=_rounded_seconds(VAD_WINDOW_SAMPLES / MODEL_SAMPLE_RATE),
        min_silence_seconds=candidate_silence_seconds,
        threshold=end_of_turn_threshold,
        speech_segments=speech_segments,
        end_of_turn_events=end_of_turn_events,
    )


@lru_cache(maxsize=1)
def _load_livekit_v1_mini_inference() -> LiveKitV1MiniInference:
    return LiveKitV1MiniInference(
        vad_model=VAD(),
        end_of_turn_model=EOT(),
    )


def _read_16khz_int16_audio(wave_path: Path) -> NormalizedPcmAudio:
    audio = read_mono_wave_audio(wave_path=wave_path)
    if audio.samples.size == 0:
        raise ValueError(
            "Cannot run LiveKit Turn Detector v1-mini on an audio file with no frames."
        )

    maximum_amplitude = float((1 << (audio.sample_width * 8 - 1)) - 1)
    normalized_samples = np.clip(audio.samples / maximum_amplitude, -1.0, 1.0).astype(np.float32)
    resampled_samples = _resample_audio(
        samples=normalized_samples,
        source_sample_rate=audio.sample_rate,
        target_sample_rate=MODEL_SAMPLE_RATE,
    )
    pcm_samples = np.clip(resampled_samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return NormalizedPcmAudio(
        samples=pcm_samples,
        sample_rate=MODEL_SAMPLE_RATE,
        duration_seconds=pcm_samples.size / MODEL_SAMPLE_RATE,
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


def _speech_segments_from_vad(
    samples: NDArray[np.int16],
    sample_rate: int,
    vad_model: LiveKitVadModel,
    vad_speech_threshold: float,
    min_speech_seconds: float,
    candidate_silence_seconds: float,
) -> list[SpeechSegment]:
    speech_flags = [
        vad_model.predict(pcm=_vad_window(samples=samples, start_index=start_index))
        >= vad_speech_threshold
        for start_index in range(0, samples.size, VAD_WINDOW_SAMPLES)
    ]
    return _speech_segments_from_flags_with_pause(
        speech_flags=speech_flags,
        frame_seconds=VAD_WINDOW_SAMPLES / sample_rate,
        min_speech_seconds=min_speech_seconds,
        candidate_silence_seconds=candidate_silence_seconds,
    )


def _vad_window(samples: NDArray[np.int16], start_index: int) -> NDArray[np.int16]:
    window = samples[start_index : start_index + VAD_WINDOW_SAMPLES]
    if window.size == VAD_WINDOW_SAMPLES:
        return window
    return np.pad(window, (0, VAD_WINDOW_SAMPLES - window.size), mode="constant").astype(np.int16)


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


def _end_of_turn_events(
    samples: NDArray[np.int16],
    sample_rate: int,
    candidate_pauses: list[CandidatePause],
    end_of_turn_model: LiveKitEndOfTurnModel,
    end_of_turn_threshold: float,
) -> list[EndOfTurnEvent]:
    end_of_turn_events: list[EndOfTurnEvent] = []
    for candidate_pause in candidate_pauses:
        window = _model_window(
            samples=samples,
            sample_rate=sample_rate,
            end_seconds=candidate_pause.evaluate_seconds,
        )
        completion_probability = end_of_turn_model.predict(pcm=window)
        if completion_probability >= end_of_turn_threshold:
            end_of_turn_events.append(
                EndOfTurnEvent(
                    time_seconds=candidate_pause.evaluate_seconds,
                    speech_start_seconds=candidate_pause.speech_start_seconds,
                    speech_end_seconds=candidate_pause.speech_end_seconds,
                    silence_seconds=candidate_pause.following_silence_seconds,
                )
            )
    return end_of_turn_events


def _model_window(
    samples: NDArray[np.int16],
    sample_rate: int,
    end_seconds: float,
) -> NDArray[np.int16]:
    end_index = min(samples.size, round(end_seconds * sample_rate))
    start_index = max(0, end_index - EOT_MAX_SAMPLES)
    window = samples[start_index:end_index]
    if window.size == 0:
        raise ValueError("Cannot run LiveKit Turn Detector v1-mini on an empty model window.")
    return window


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, 6)
