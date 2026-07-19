from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.local.analyses.asr.models import AsrModelMode
from app.local.analyses.asr.service import IngestedAsrTranscriptSource, analyze_asr
from app.local.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.local.analyses.end_of_turn.cache import LeastRecentlyUsedCache
from app.local.analyses.end_of_turn.conversation_scoring import (
    ConversationScoringConfig,
    score_conversation,
)
from app.local.analyses.end_of_turn.detectors.naive_vad import run_naive_vad_floor
from app.local.analyses.end_of_turn.detectors.two_speaker_annotation import (
    INTERNAL_PAUSE_SECONDS,
    OTHER_SPEAKER,
    TARGET_SPEAKER,
    TURN_GAP_SECONDS,
    TranscriptTurn,
    TranscriptWord,
)
from app.local.analyses.end_of_turn.service import BaselineResult, SpeechSegment
from app.local.asr.client import HttpRemoteAsrClient
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.asr.repository import AsrTranscriptRepository
from app.local.asr.service import AsrTranscriptCache, RemoteAsrClientFactory
from app.local.asr.transcript import SpeakerTrack, Word
from app.local.config import COMPUTE_TOKEN, DATABASE_URL, REMOTE_ASR_ENDPOINT_URL
from app.shared.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS

ASR_MODEL_MODE = AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED
TRANSCRIPT_PAIR_CACHE_SIZE = 20
SPEECH_SEGMENT_GAP_SECONDS = 0.5
SCORING_CONFIG = ConversationScoringConfig()


@dataclass(frozen=True)
class AsrTranscriptPairCacheKey:
    speaker1_path: Path
    speaker1_modified_ns: int
    speaker1_size_bytes: int
    speaker2_path: Path
    speaker2_modified_ns: int
    speaker2_size_bytes: int


@dataclass(frozen=True)
class AsrTranscriptPair:
    turns: tuple[TranscriptTurn, ...]
    analysis_end_seconds: float


class AsrTwoSpeakerTranscriptSource:
    def __init__(
        self,
        cache: AsrTranscriptCache,
        ingested_transcript_source: IngestedAsrTranscriptSource,
        remote_client_factory: RemoteAsrClientFactory,
    ) -> None:
        self.cache = cache
        self.ingested_transcript_source = ingested_transcript_source
        self.remote_client_factory = remote_client_factory
        self._transcript_pair_cache = LeastRecentlyUsedCache[
            AsrTranscriptPairCacheKey, AsrTranscriptPair
        ](capacity=TRANSCRIPT_PAIR_CACHE_SIZE)
        self._load_lock = Lock()

    def load(self, speaker1_path: Path) -> AsrTranscriptPair:
        speaker2_path = _speaker2_path(speaker1_path=speaker1_path)
        cache_key = _transcript_pair_cache_key(
            speaker1_path=speaker1_path,
            speaker2_path=speaker2_path,
        )
        with self._load_lock:
            cached_pair = self._transcript_pair_cache.get(cache_key=cache_key)
            if cached_pair is not None:
                return cached_pair
            transcript_pair = self._load_uncached(
                speaker1_path=speaker1_path,
            )
            self._transcript_pair_cache.put(
                cache_key=cache_key,
                cache_value=transcript_pair,
            )
            return transcript_pair

    def _load_uncached(
        self,
        speaker1_path: Path,
    ) -> AsrTranscriptPair:
        session_identifier = _session_identifier(speaker1_path=speaker1_path)
        speaker1_response = analyze_asr(
            session_id=session_identifier,
            speaker_track=SpeakerTrack.SPEAKER1,
            selected_models=(ASR_MODEL_MODE,),
            reference_words=(),
            cache=self.cache,
            ingested_transcript_source=self.ingested_transcript_source,
            remote_client_factory=self.remote_client_factory,
        )
        speaker2_response = analyze_asr(
            session_id=session_identifier,
            speaker_track=SpeakerTrack.SPEAKER2,
            selected_models=(ASR_MODEL_MODE,),
            reference_words=(),
            cache=self.cache,
            ingested_transcript_source=self.ingested_transcript_source,
            remote_client_factory=self.remote_client_factory,
        )
        if len(speaker1_response.runs) != 1 or len(speaker2_response.runs) != 1:
            raise ValueError("ASR two-speaker annotation requires one transcript per speaker.")
        turns = merged_asr_turns(
            speaker1_words=speaker1_response.runs[0].transcription.words,
            speaker2_words=speaker2_response.runs[0].transcription.words,
            turn_gap_seconds=SPEECH_SEGMENT_GAP_SECONDS,
        )
        return AsrTranscriptPair(
            turns=tuple(turns),
            analysis_end_seconds=min(
                speaker1_response.analyzed_duration_seconds,
                speaker2_response.analyzed_duration_seconds,
                ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
            ),
        )


@dataclass(frozen=True)
class AsrTwoSpeakerAnnotationDetector:
    info: EndOfTurnDetectorInfo
    target_track: SpeakerTrack
    transcript_source: AsrTwoSpeakerTranscriptSource
    turn_gap_seconds: float
    internal_pause_seconds: float

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        transcript_pair = self.transcript_source.load(speaker1_path=speaker1_path)
        match self.target_track:
            case SpeakerTrack.SPEAKER1:
                target_speaker = TARGET_SPEAKER
                other_speaker = OTHER_SPEAKER
            case SpeakerTrack.SPEAKER2:
                target_speaker = OTHER_SPEAKER
                other_speaker = TARGET_SPEAKER
        scored_conversation = score_conversation(
            turns=list(transcript_pair.turns),
            audio_activity_segments=_audio_activity_segments(
                wave_path=(
                    speaker1_path
                    if self.target_track == SpeakerTrack.SPEAKER1
                    else _speaker2_path(speaker1_path=speaker1_path)
                )
            ),
            analysis_end_seconds=transcript_pair.analysis_end_seconds,
            target_speaker=target_speaker,
            other_speaker=other_speaker,
            config=SCORING_CONFIG,
        )
        return BaselineResult(
            name=self.info.mode.value,
            description=self.info.description,
            frame_seconds=self.internal_pause_seconds,
            min_silence_seconds=self.turn_gap_seconds,
            threshold=SCORING_CONFIG.keep_playing_threshold,
            speech_segments=scored_conversation.speech_segments,
            pause_spans=scored_conversation.pause_spans,
            backchannel_spans=scored_conversation.backchannel_spans,
            end_of_turn_events=scored_conversation.end_of_turn_events,
            interruption_events=scored_conversation.interruption_events,
            segment_hypotheses=scored_conversation.segment_hypotheses,
            connection_hypotheses=scored_conversation.connection_hypotheses,
        )


def asr_two_speaker_annotation_detectors() -> tuple[
    AsrTwoSpeakerAnnotationDetector,
    AsrTwoSpeakerAnnotationDetector,
]:
    transcript_source = AsrTwoSpeakerTranscriptSource(
        cache=AsrTranscriptRepository(database_url=DATABASE_URL),
        ingested_transcript_source=FullRecordingAsrRepository(database_url=DATABASE_URL),
        remote_client_factory=remote_asr_client,
    )
    return (
        _asr_two_speaker_annotation_detector(
            mode=EndOfTurnDetectorMode.ASR_TWO_SPEAKER_ANNOTATION,
            target_track=SpeakerTrack.SPEAKER1,
            label="ASR two-speaker annotation - Speaker 1",
            transcript_source=transcript_source,
        ),
        _asr_two_speaker_annotation_detector(
            mode=EndOfTurnDetectorMode.ASR_TWO_SPEAKER_ANNOTATION_SPEAKER2,
            target_track=SpeakerTrack.SPEAKER2,
            label="ASR two-speaker annotation - Speaker 2",
            transcript_source=transcript_source,
        ),
    )


def asr_two_speaker_annotation_detector() -> AsrTwoSpeakerAnnotationDetector:
    return asr_two_speaker_annotation_detectors()[0]


def _asr_two_speaker_annotation_detector(
    mode: EndOfTurnDetectorMode,
    target_track: SpeakerTrack,
    label: str,
    transcript_source: AsrTwoSpeakerTranscriptSource | None = None,
) -> AsrTwoSpeakerAnnotationDetector:
    source = transcript_source or AsrTwoSpeakerTranscriptSource(
        cache=AsrTranscriptRepository(database_url=DATABASE_URL),
        ingested_transcript_source=FullRecordingAsrRepository(database_url=DATABASE_URL),
        remote_client_factory=remote_asr_client,
    )
    return AsrTwoSpeakerAnnotationDetector(
        info=EndOfTurnDetectorInfo(
            mode=mode,
            label=label,
            description=(
                f"{label} with keep-playing, turn, interruption, pause, and merge confidence "
                "scores over cached crosstalk-filtered Parakeet + Canary transcripts and "
                "energy activity."
            ),
        ),
        target_track=target_track,
        transcript_source=source,
        turn_gap_seconds=TURN_GAP_SECONDS,
        internal_pause_seconds=INTERNAL_PAUSE_SECONDS,
    )


def remote_asr_client() -> HttpRemoteAsrClient:
    return HttpRemoteAsrClient(
        endpoint_url=REMOTE_ASR_ENDPOINT_URL,
        api_key=COMPUTE_TOKEN,
    )


def _audio_activity_segments(wave_path: Path) -> list[SpeechSegment]:
    return run_naive_vad_floor(
        wave_path=wave_path,
        result_name="asr_audio_activity",
        description="Energy activity used to expose audio omitted by ASR.",
        frame_seconds=0.03,
        min_speech_seconds=0.12,
        min_silence_seconds=0.4,
    ).speech_segments


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


def _speaker2_path(speaker1_path: Path) -> Path:
    session_identifier = _session_identifier(speaker1_path=speaker1_path)
    speaker2_path = speaker1_path.with_name(f"{session_identifier}_speaker2.wav")
    if not speaker2_path.is_file():
        raise ValueError(f"Missing speaker 2 WAV beside speaker 1 WAV: {speaker2_path}")
    return speaker2_path


def _transcript_pair_cache_key(
    speaker1_path: Path,
    speaker2_path: Path,
) -> AsrTranscriptPairCacheKey:
    speaker1_status = speaker1_path.stat()
    speaker2_status = speaker2_path.stat()
    return AsrTranscriptPairCacheKey(
        speaker1_path=speaker1_path,
        speaker1_modified_ns=speaker1_status.st_mtime_ns,
        speaker1_size_bytes=speaker1_status.st_size,
        speaker2_path=speaker2_path,
        speaker2_modified_ns=speaker2_status.st_mtime_ns,
        speaker2_size_bytes=speaker2_status.st_size,
    )
