from __future__ import annotations

import time
from dataclasses import dataclass

from app.asr_quality.alignment import align_words
from app.asr_quality.schemas import AlignmentOperation, SpeakerTrack, TranscriptionResult, Word

MERGED_CONSENSUS_MODEL_NAME = "merged_consensus"
MERGED_CONSENSUS_IDENTIFIER = "voice-light/asr-consensus-v1"
PARAKEET_CANARY_CONSENSUS_MODEL_NAME = "parakeet_canary_consensus"
PARAKEET_CANARY_CONSENSUS_IDENTIFIER = "voice-light/parakeet-canary-consensus-v3"


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
    words = parakeet_canary_union_words(
        parakeet_words=parakeet.words,
        canary_words=canary.words,
    )
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
            "merge_strategy": "parakeet_priority_timing_union",
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


def parakeet_canary_union_words(
    parakeet_words: tuple[Word, ...], canary_words: tuple[Word, ...]
) -> tuple[Word, ...]:
    canary_only_words = tuple(
        canary_word
        for canary_word in canary_words
        if is_timestamped(canary_word)
        and not any(
            words_overlap(first=canary_word, second=parakeet_word)
            for parakeet_word in parakeet_words
        )
    )
    return sorted_words_by_start((*parakeet_words, *canary_only_words))


def is_timestamped(word: Word) -> bool:
    return word.start_seconds is not None and word.end_seconds is not None


def words_overlap(first: Word, second: Word) -> bool:
    if not is_timestamped(first) or not is_timestamped(second):
        return False
    assert first.start_seconds is not None
    assert first.end_seconds is not None
    assert second.start_seconds is not None
    assert second.end_seconds is not None
    return first.start_seconds <= second.end_seconds and first.end_seconds >= second.start_seconds


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
