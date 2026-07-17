from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from app.local.asr.transcript import Word
from app.shared.audio import AudioTrack
from app.shared.quality import AudioMetadata


class SynchronizationAlignmentOrigin(StrEnum):
    REVIEWED = "reviewed"
    UNREVIEWED_ZERO = "unreviewed_zero"


@dataclass(frozen=True)
class SynchronizationAlignment:
    speaker2_shift_seconds: float
    origin: SynchronizationAlignmentOrigin


@dataclass(frozen=True)
class AlignedAudioPair:
    speaker1: AudioTrack
    speaker2: AudioTrack

    @property
    def duration_seconds(self) -> float:
        return self.speaker1.metadata.duration_seconds


def align_audio_tracks(
    speaker1: AudioTrack,
    speaker2: AudioTrack,
    speaker2_shift_seconds: float,
) -> AlignedAudioPair:
    if speaker1.metadata.sample_rate != speaker2.metadata.sample_rate:
        raise ValueError("Synchronization requires matching audio sample rates.")
    sample_rate = speaker1.metadata.sample_rate
    shift_samples = round(speaker2_shift_seconds * sample_rate)
    shifted_speaker2_samples = (
        speaker2.samples[-shift_samples:]
        if shift_samples < 0
        else (
            np.pad(speaker2.samples, ((shift_samples, 0), (0, 0)))
            if shift_samples > 0
            else speaker2.samples
        )
    )
    shared_sample_count = max(len(speaker1.samples), len(shifted_speaker2_samples))
    return AlignedAudioPair(
        speaker1=_audio_track_with_sample_count(
            audio=speaker1,
            samples=speaker1.samples,
            sample_count=shared_sample_count,
        ),
        speaker2=_audio_track_with_sample_count(
            audio=speaker2,
            samples=shifted_speaker2_samples,
            sample_count=shared_sample_count,
        ),
    )


def shift_words(
    words: tuple[Word, ...],
    shift_seconds: float,
    timeline_duration_seconds: float,
) -> tuple[Word, ...]:
    shifted_words: list[Word] = []
    for word in words:
        if word.start_seconds is None or word.end_seconds is None:
            raise ValueError(f"Full-recording ASR word is missing timestamps: {word.text}")
        shifted_start = word.start_seconds + shift_seconds
        shifted_end = word.end_seconds + shift_seconds
        if shifted_end <= 0.0 or shifted_start >= timeline_duration_seconds:
            continue
        clipped_start = max(0.0, shifted_start)
        clipped_end = min(timeline_duration_seconds, shifted_end)
        if clipped_end <= clipped_start:
            continue
        shifted_words.append(
            Word(
                text=word.text,
                start_seconds=clipped_start,
                end_seconds=clipped_end,
                confidence=word.confidence,
            )
        )
    return tuple(shifted_words)


def _audio_track_with_sample_count(
    audio: AudioTrack,
    samples: NDArray[np.float32],
    sample_count: int,
) -> AudioTrack:
    missing_sample_count = sample_count - len(samples)
    assert missing_sample_count >= 0
    padded_samples = (
        np.pad(samples, ((0, missing_sample_count), (0, 0))) if missing_sample_count else samples
    )
    return AudioTrack(
        samples=padded_samples.astype(np.float32, copy=False),
        metadata=AudioMetadata(
            duration_seconds=sample_count / audio.metadata.sample_rate,
            sample_rate=audio.metadata.sample_rate,
            channels=padded_samples.shape[1],
            sample_count=sample_count,
        ),
    )
