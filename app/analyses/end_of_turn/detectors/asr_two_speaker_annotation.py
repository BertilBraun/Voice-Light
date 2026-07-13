from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.analyses.asr.models import AsrModelMode
from app.analyses.asr.service import analyze_asr
from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.detectors.two_speaker_annotation import (
    INTERNAL_PAUSE_SECONDS,
    OTHER_SPEAKER,
    TARGET_SPEAKER,
    TURN_GAP_SECONDS,
    TranscriptTurn,
    TranscriptWord,
    analyze_two_speaker_turns,
)
from app.analyses.end_of_turn.service import BaselineResult
from app.asr.client import HttpRemoteAsrClient
from app.asr.repository import AsrTranscriptRepository
from app.asr.service import AsrTranscriptCache, RemoteAsrClientFactory
from app.asr.transcript import SpeakerTrack, Word
from app.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS
from app.config import COMPUTE_TOKEN, DATABASE_URL, REMOTE_ASR_ENDPOINT_URL

ASR_MODEL_MODE = AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED


@dataclass(frozen=True)
class AsrTwoSpeakerAnnotationDetector:
    info: EndOfTurnDetectorInfo
    turn_gap_seconds: float
    internal_pause_seconds: float
    cache: AsrTranscriptCache
    remote_client_factory: RemoteAsrClientFactory

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        session_identifier = _session_identifier(speaker1_path=speaker1_path)
        speaker1_response = analyze_asr(
            session_id=session_identifier,
            speaker_track=SpeakerTrack.SPEAKER1,
            selected_models=(ASR_MODEL_MODE,),
            reference_words=(),
            cache=self.cache,
            remote_client_factory=self.remote_client_factory,
        )
        speaker2_response = analyze_asr(
            session_id=session_identifier,
            speaker_track=SpeakerTrack.SPEAKER2,
            selected_models=(ASR_MODEL_MODE,),
            reference_words=(),
            cache=self.cache,
            remote_client_factory=self.remote_client_factory,
        )
        if len(speaker1_response.runs) != 1 or len(speaker2_response.runs) != 1:
            raise ValueError("ASR two-speaker annotation requires one transcript per speaker.")

        turns = merged_asr_turns(
            speaker1_words=speaker1_response.runs[0].transcription.words,
            speaker2_words=speaker2_response.runs[0].transcription.words,
            turn_gap_seconds=self.turn_gap_seconds,
        )
        analysis_end_seconds = min(
            speaker1_response.analyzed_duration_seconds,
            speaker2_response.analyzed_duration_seconds,
            ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
        )
        return analyze_two_speaker_turns(
            speaker1_path=speaker1_path,
            turns=turns,
            analysis_end_seconds=analysis_end_seconds,
            result_name=self.info.mode.value,
            description=self.info.description,
            turn_gap_seconds=self.turn_gap_seconds,
            internal_pause_seconds=self.internal_pause_seconds,
        )


def asr_two_speaker_annotation_detector() -> AsrTwoSpeakerAnnotationDetector:
    return AsrTwoSpeakerAnnotationDetector(
        info=EndOfTurnDetectorInfo(
            mode=EndOfTurnDetectorMode.ASR_TWO_SPEAKER_ANNOTATION,
            label="ASR two-speaker annotation",
            description=(
                "Speaker 1 labels from cached Parakeet + Canary timing-union transcripts "
                "for both channels, independently crosstalk-filtered before merging."
            ),
        ),
        turn_gap_seconds=TURN_GAP_SECONDS,
        internal_pause_seconds=INTERNAL_PAUSE_SECONDS,
        cache=AsrTranscriptRepository(database_url=DATABASE_URL),
        remote_client_factory=remote_asr_client,
    )


def remote_asr_client() -> HttpRemoteAsrClient:
    return HttpRemoteAsrClient(
        endpoint_url=REMOTE_ASR_ENDPOINT_URL,
        api_key=COMPUTE_TOKEN,
    )


def merged_asr_turns(
    speaker1_words: tuple[Word, ...],
    speaker2_words: tuple[Word, ...],
    turn_gap_seconds: float,
) -> list[TranscriptTurn]:
    turns = [
        *_speaker_turns(
            words=speaker1_words,
            speaker=TARGET_SPEAKER,
            turn_gap_seconds=turn_gap_seconds,
        ),
        *_speaker_turns(
            words=speaker2_words,
            speaker=OTHER_SPEAKER,
            turn_gap_seconds=turn_gap_seconds,
        ),
    ]
    return sorted(turns, key=lambda turn: (turn.start_seconds, turn.end_seconds, turn.speaker))


def _speaker_turns(
    words: tuple[Word, ...],
    speaker: str,
    turn_gap_seconds: float,
) -> list[TranscriptTurn]:
    timed_words = sorted(
        (_transcript_word(word=word) for word in words),
        key=lambda word: (word.start_seconds, word.end_seconds),
    )
    if not timed_words:
        return []

    grouped_words: list[list[TranscriptWord]] = [[timed_words[0]]]
    for word in timed_words[1:]:
        previous_word = grouped_words[-1][-1]
        if word.start_seconds - previous_word.end_seconds > turn_gap_seconds:
            grouped_words.append([word])
            continue
        grouped_words[-1].append(word)
    return [_transcript_turn(speaker=speaker, words=group) for group in grouped_words]


def _transcript_word(word: Word) -> TranscriptWord:
    if word.start_seconds is None or word.end_seconds is None:
        raise ValueError(f"ASR word is missing timestamps: {word.text}")
    if word.end_seconds < word.start_seconds:
        raise ValueError(f"ASR word end precedes start: {word.text}")
    return TranscriptWord(
        text=word.text,
        start_seconds=word.start_seconds,
        end_seconds=word.end_seconds,
    )


def _transcript_turn(speaker: str, words: list[TranscriptWord]) -> TranscriptTurn:
    return TranscriptTurn(
        speaker=speaker,
        text=" ".join(word.text for word in words),
        start_seconds=words[0].start_seconds,
        end_seconds=max(word.end_seconds for word in words),
        words=words,
    )


def _session_identifier(speaker1_path: Path) -> str:
    suffix = "_speaker1"
    if not speaker1_path.stem.endswith(suffix):
        raise ValueError(f"Expected a speaker 1 WAV path ending in `{suffix}.wav`: {speaker1_path}")
    session_identifier = speaker1_path.stem[: -len(suffix)]
    if not session_identifier:
        raise ValueError(f"Could not derive session identifier from WAV path: {speaker1_path}")
    return session_identifier
