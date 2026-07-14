from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.local.analyses.end_of_turn.cache import LeastRecentlyUsedCache, WaveformCacheKey
from app.local.data.transcripts import TranscriptTurn, transcript_turn_to_json
from app.shared.audio.wav import read_mono_wave_audio

WAVEFORM_CACHE_SIZE = 40


@dataclass(frozen=True)
class WaveformEnvelope:
    sample_rate: int
    duration_seconds: float
    samples_per_bin: int
    minimums: list[float]
    maximums: list[float]


@dataclass(frozen=True)
class SpeechSegment:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class EndOfTurnEvent:
    time_seconds: float
    speech_start_seconds: float
    speech_end_seconds: float
    silence_seconds: float


@dataclass(frozen=True)
class PauseSpan:
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class BackchannelSpan:
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    text: str


@dataclass(frozen=True)
class InterruptionEvent:
    time_seconds: float
    confidence: float
    interrupted_speaker: str
    interrupting_speaker: str
    text: str


@dataclass(frozen=True)
class SegmentHypothesis:
    start_seconds: float
    end_seconds: float
    text: str
    backchannel_confidence: float
    turn_confidence: float
    interruption_confidence: float


@dataclass(frozen=True)
class ConnectionHypothesis:
    earlier_end_seconds: float
    later_start_seconds: float
    gap_seconds: float
    pause_confidence: float
    merge_confidence: float


@dataclass(frozen=True)
class BaselineResult:
    name: str
    description: str
    frame_seconds: float
    min_silence_seconds: float
    threshold: float
    speech_segments: list[SpeechSegment]
    pause_spans: list[PauseSpan]
    backchannel_spans: list[BackchannelSpan]
    end_of_turn_events: list[EndOfTurnEvent]
    interruption_events: list[InterruptionEvent] = field(default_factory=list)
    segment_hypotheses: list[SegmentHypothesis] = field(default_factory=list)
    connection_hypotheses: list[ConnectionHypothesis] = field(default_factory=list)


@dataclass(frozen=True)
class AudioAnalysis:
    speaker1_waveform: WaveformEnvelope
    speaker2_waveform: WaveformEnvelope
    transcript_turns: list[TranscriptTurn]
    baseline_results: list[BaselineResult]


_WAVEFORM_ENVELOPE_CACHE = LeastRecentlyUsedCache[WaveformCacheKey, WaveformEnvelope](
    capacity=WAVEFORM_CACHE_SIZE
)


def waveform_to_json(waveform: WaveformEnvelope) -> dict[str, object]:
    return asdict(waveform)


def baseline_to_json(baseline_result: BaselineResult) -> dict[str, object]:
    return asdict(baseline_result)


def analysis_to_json(audio_analysis: AudioAnalysis) -> dict[str, object]:
    return {
        "speaker1_waveform": waveform_to_json(audio_analysis.speaker1_waveform),
        "speaker2_waveform": waveform_to_json(audio_analysis.speaker2_waveform),
        "transcript_turns": [
            transcript_turn_to_json(transcript_turn)
            for transcript_turn in audio_analysis.transcript_turns
        ],
        "baseline_results": [
            baseline_to_json(baseline_result) for baseline_result in audio_analysis.baseline_results
        ],
    }


def build_waveform_envelope(
    wave_path: Path,
    target_bins: int = 60000,
) -> WaveformEnvelope:
    cache_key = _waveform_cache_key(
        wave_path=wave_path,
        target_bins=target_bins,
    )
    cached_waveform = _WAVEFORM_ENVELOPE_CACHE.get(cache_key=cache_key)
    if cached_waveform is not None:
        return cached_waveform

    audio = read_mono_wave_audio(wave_path=wave_path)
    samples_per_bin = max(1, math.ceil(audio.frame_count / target_bins))
    maximum_amplitude = float((1 << (audio.sample_width * 8 - 1)) - 1)
    minimums: list[float] = []
    maximums: list[float] = []

    for start_index in range(0, audio.frame_count, samples_per_bin):
        samples = audio.samples[start_index : start_index + samples_per_bin]
        minimum = float(np.min(samples))
        maximum = float(np.max(samples))
        minimums.append(max(-1.0, minimum / maximum_amplitude))
        maximums.append(min(1.0, maximum / maximum_amplitude))

    waveform_envelope = WaveformEnvelope(
        sample_rate=audio.sample_rate,
        duration_seconds=audio.duration_seconds,
        samples_per_bin=samples_per_bin,
        minimums=minimums,
        maximums=maximums,
    )
    _WAVEFORM_ENVELOPE_CACHE.put(cache_key=cache_key, cache_value=waveform_envelope)
    return waveform_envelope


def analyze_session_audio(
    speaker1_path: Path,
    speaker2_path: Path,
    transcript_turns: list[TranscriptTurn],
    baseline_results: list[BaselineResult],
) -> AudioAnalysis:
    speaker1_waveform = build_waveform_envelope(wave_path=speaker1_path)
    speaker2_waveform = build_waveform_envelope(wave_path=speaker2_path)
    return AudioAnalysis(
        speaker1_waveform=speaker1_waveform,
        speaker2_waveform=speaker2_waveform,
        transcript_turns=transcript_turns,
        baseline_results=baseline_results,
    )


def _waveform_cache_key(
    wave_path: Path,
    target_bins: int,
) -> WaveformCacheKey:
    file_status = wave_path.stat()
    return WaveformCacheKey(
        wave_path=wave_path,
        target_bins=target_bins,
        modified_ns=file_status.st_mtime_ns,
        size_bytes=file_status.st_size,
    )
