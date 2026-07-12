from __future__ import annotations

import time
from dataclasses import dataclass

from app.asr_quality.alignment import align_words
from app.asr_quality.schemas import AlignmentOperation, SpeakerTrack, TranscriptionResult, Word

MERGED_CONSENSUS_MODEL_NAME = "merged_consensus"
MERGED_CONSENSUS_IDENTIFIER = "voice-light/asr-consensus-v1"
PARAKEET_CANARY_CONSENSUS_MODEL_NAME = "parakeet_canary_consensus"
PARAKEET_CANARY_CONSENSUS_IDENTIFIER = "voice-light/parakeet-canary-consensus-v1"
PAIRWISE_TIMESTAMP_TOLERANCE_SECONDS = 0.2


@dataclass(frozen=True)
class TranscriptToMerge:
    model_name: str
    words: tuple[Word, ...]


def merged_consensus_transcription(
    audio_path: str,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    transcripts: tuple[TranscriptToMerge, ...],
) -> TranscriptionResult:
    if len(transcripts) < 2:
        raise ValueError("Merged ASR consensus requires at least two transcripts.")
    merge_start = time.perf_counter()
    words = progressively_merged_words(transcripts=transcripts)
    processing_time_seconds = time.perf_counter() - merge_start
    return TranscriptionResult(
        model_name=MERGED_CONSENSUS_MODEL_NAME,
        audio_path=audio_path,
        track=speaker_track,
        audio_duration_seconds=audio_duration_seconds,
        processing_time_seconds=processing_time_seconds,
        words=words,
        raw_output={
            "source_models": ",".join(transcript.model_name for transcript in transcripts),
            "word_count": len(words),
        },
        model_identifier=MERGED_CONSENSUS_IDENTIFIER,
        package_versions={},
        model_loading_time_seconds=0.0,
        inference_time_seconds=processing_time_seconds,
        real_time_factor=processing_time_seconds / audio_duration_seconds
        if audio_duration_seconds > 0.0
        else None,
        peak_gpu_memory_mb=None,
        error=None,
    )


def progressively_merged_words(transcripts: tuple[TranscriptToMerge, ...]) -> tuple[Word, ...]:
    merged = transcripts[0].words
    for transcript in transcripts[1:]:
        merged = merged_words(primary=merged, secondary=transcript.words)
    return merged


def parakeet_canary_consensus_transcription(
    audio_path: str,
    speaker_track: SpeakerTrack,
    audio_duration_seconds: float,
    parakeet: TranscriptToMerge,
    canary: TranscriptToMerge,
) -> TranscriptionResult:
    merge_start = time.perf_counter()
    words = agreeing_conservative_words(primary=parakeet.words, secondary=canary.words)
    processing_time_seconds = time.perf_counter() - merge_start
    return TranscriptionResult(
        model_name=PARAKEET_CANARY_CONSENSUS_MODEL_NAME,
        audio_path=audio_path,
        track=speaker_track,
        audio_duration_seconds=audio_duration_seconds,
        processing_time_seconds=processing_time_seconds,
        words=words,
        raw_output={
            "source_models": f"{parakeet.model_name},{canary.model_name}",
            "timestamp_tolerance_seconds": PAIRWISE_TIMESTAMP_TOLERANCE_SECONDS,
            "word_count": len(words),
        },
        model_identifier=PARAKEET_CANARY_CONSENSUS_IDENTIFIER,
        package_versions={},
        model_loading_time_seconds=0.0,
        inference_time_seconds=processing_time_seconds,
        real_time_factor=processing_time_seconds / audio_duration_seconds
        if audio_duration_seconds > 0.0
        else None,
        peak_gpu_memory_mb=None,
        error=None,
    )


def merged_words(primary: tuple[Word, ...], secondary: tuple[Word, ...]) -> tuple[Word, ...]:
    alignment = align_words(primary, secondary)
    words: list[Word] = []
    for aligned_word in alignment:
        match aligned_word.operation:
            case AlignmentOperation.EQUAL:
                assert aligned_word.reference is not None
                assert aligned_word.prediction is not None
                words.append(
                    average_word_timestamps(aligned_word.reference, aligned_word.prediction)
                )
            case AlignmentOperation.SUBSTITUTE:
                assert aligned_word.reference is not None
                assert aligned_word.prediction is not None
                words.extend(
                    sorted_words_by_start((aligned_word.reference, aligned_word.prediction))
                )
            case AlignmentOperation.DELETE:
                assert aligned_word.reference is not None
                words.append(aligned_word.reference)
            case AlignmentOperation.INSERT:
                assert aligned_word.prediction is not None
                words.append(aligned_word.prediction)
    return tuple(words)


def agreeing_conservative_words(
    primary: tuple[Word, ...], secondary: tuple[Word, ...]
) -> tuple[Word, ...]:
    words: list[Word] = []
    for aligned_word in align_words(primary, secondary):
        if aligned_word.operation is not AlignmentOperation.EQUAL:
            continue
        assert aligned_word.reference is not None
        assert aligned_word.prediction is not None
        if not timestamps_agree_within_tolerance(
            primary=aligned_word.reference,
            secondary=aligned_word.prediction,
        ):
            continue
        words.append(
            intersecting_word_timestamps(
                primary=aligned_word.reference,
                secondary=aligned_word.prediction,
            )
        )
    return tuple(words)


def timestamps_agree_within_tolerance(primary: Word, secondary: Word) -> bool:
    if (
        primary.start_seconds is None
        or primary.end_seconds is None
        or secondary.start_seconds is None
        or secondary.end_seconds is None
    ):
        return False
    return (
        abs(primary.start_seconds - secondary.start_seconds) <= PAIRWISE_TIMESTAMP_TOLERANCE_SECONDS
        and abs(primary.end_seconds - secondary.end_seconds) <= PAIRWISE_TIMESTAMP_TOLERANCE_SECONDS
        and max(primary.start_seconds, secondary.start_seconds)
        <= min(primary.end_seconds, secondary.end_seconds)
    )


def intersecting_word_timestamps(primary: Word, secondary: Word) -> Word:
    assert primary.start_seconds is not None
    assert primary.end_seconds is not None
    assert secondary.start_seconds is not None
    assert secondary.end_seconds is not None
    start_seconds = max(primary.start_seconds, secondary.start_seconds)
    end_seconds = min(primary.end_seconds, secondary.end_seconds)
    assert start_seconds <= end_seconds
    return Word(
        text=primary.text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        confidence=average_optional_seconds(primary.confidence, secondary.confidence),
    )


def average_word_timestamps(primary: Word, secondary: Word) -> Word:
    return Word(
        text=primary.text,
        start_seconds=average_optional_seconds(primary.start_seconds, secondary.start_seconds),
        end_seconds=average_optional_seconds(primary.end_seconds, secondary.end_seconds),
        confidence=average_optional_seconds(primary.confidence, secondary.confidence),
    )


def average_optional_seconds(primary: float | None, secondary: float | None) -> float | None:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    return (primary + secondary) / 2.0


def sorted_words_by_start(words: tuple[Word, ...]) -> tuple[Word, ...]:
    return tuple(
        sorted(
            words,
            key=lambda word: word.start_seconds if word.start_seconds is not None else float("inf"),
        )
    )
