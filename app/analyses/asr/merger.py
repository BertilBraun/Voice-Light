from __future__ import annotations

import time
from dataclasses import dataclass

from app.asr_quality.alignment import align_words
from app.asr_quality.schemas import AlignmentOperation, SpeakerTrack, TranscriptionResult, Word

MERGED_CONSENSUS_MODEL_NAME = "merged_consensus"
MERGED_CONSENSUS_IDENTIFIER = "voice-light/asr-consensus-v1"
PARAKEET_CANARY_CONSENSUS_MODEL_NAME = "parakeet_canary_consensus"
PARAKEET_CANARY_CONSENSUS_IDENTIFIER = "voice-light/parakeet-canary-consensus-v2"
CANARY_SPEECH_BLOCK_GAP_SECONDS = 0.3


@dataclass(frozen=True)
class TranscriptToMerge:
    model_name: str
    words: tuple[Word, ...]


@dataclass(frozen=True)
class SpeechBlock:
    start_seconds: float
    end_seconds: float


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
    words = parakeet_words_in_canary_speech_blocks(
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
            "canary_speech_block_gap_seconds": CANARY_SPEECH_BLOCK_GAP_SECONDS,
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


def parakeet_words_in_canary_speech_blocks(
    parakeet_words: tuple[Word, ...], canary_words: tuple[Word, ...]
) -> tuple[Word, ...]:
    words: list[Word] = []
    speech_blocks = speech_blocks_from_words(canary_words)
    for parakeet_word in parakeet_words:
        speech_block = overlapping_speech_block(word=parakeet_word, speech_blocks=speech_blocks)
        if speech_block is None:
            continue
        words.append(clipped_word_to_speech_block(word=parakeet_word, speech_block=speech_block))
    return tuple(words)


def speech_blocks_from_words(words: tuple[Word, ...]) -> tuple[SpeechBlock, ...]:
    timestamped_words = tuple(
        sorted(
            (
                word
                for word in words
                if word.start_seconds is not None and word.end_seconds is not None
            ),
            key=lambda word: word.start_seconds,
        )
    )
    speech_blocks: list[SpeechBlock] = []
    for word in timestamped_words:
        assert word.start_seconds is not None
        assert word.end_seconds is not None
        previous_block = speech_blocks[-1] if speech_blocks else None
        if (
            previous_block is not None
            and word.start_seconds - previous_block.end_seconds <= CANARY_SPEECH_BLOCK_GAP_SECONDS
        ):
            speech_blocks[-1] = SpeechBlock(
                start_seconds=previous_block.start_seconds,
                end_seconds=max(previous_block.end_seconds, word.end_seconds),
            )
            continue
        speech_blocks.append(
            SpeechBlock(start_seconds=word.start_seconds, end_seconds=word.end_seconds)
        )
    return tuple(speech_blocks)


def overlapping_speech_block(
    word: Word, speech_blocks: tuple[SpeechBlock, ...]
) -> SpeechBlock | None:
    if word.start_seconds is None or word.end_seconds is None:
        return None
    for speech_block in speech_blocks:
        if (
            word.start_seconds <= speech_block.end_seconds
            and word.end_seconds >= speech_block.start_seconds
        ):
            return speech_block
    return None


def clipped_word_to_speech_block(word: Word, speech_block: SpeechBlock) -> Word:
    assert word.start_seconds is not None
    assert word.end_seconds is not None
    start_seconds = max(word.start_seconds, speech_block.start_seconds)
    end_seconds = min(word.end_seconds, speech_block.end_seconds)
    assert start_seconds <= end_seconds
    return Word(
        text=word.text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        confidence=word.confidence,
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
