from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

TURN_GAP_SECONDS = 1.5
INTERNAL_PAUSE_SECONDS = 0.5
ROUND_SECONDS_DIGITS = 6


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class SpeakerTranscript:
    speaker: str
    words: list[TranscriptWord]


@dataclass(frozen=True)
class SessionTranscript:
    name: str
    duration_seconds: float
    speaker_transcripts: list[SpeakerTranscript]


@dataclass(frozen=True)
class SpeakerMetadataLocation:
    metadata_path: Path
    speaker_name: str


@dataclass(frozen=True)
class TranscriptGapDetector:
    info: EndOfTurnDetectorInfo
    turn_gap_seconds: float
    internal_pause_seconds: float

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        metadata_location = _speaker_metadata_location(wave_path=speaker1_path)
        session_transcript = _read_session_transcript(metadata_path=metadata_location.metadata_path)
        analysis_end_seconds = min(
            session_transcript.duration_seconds,
            ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
        )
        words = _speaker_words(
            session_transcript=session_transcript,
            speaker_name=metadata_location.speaker_name,
            analysis_end_seconds=analysis_end_seconds,
        )
        speech_segments = _turn_segments(
            words=words,
            turn_gap_seconds=self.turn_gap_seconds,
        )
        end_of_turn_events = _end_of_turn_events(
            speech_segments=speech_segments,
            analysis_end_seconds=analysis_end_seconds,
            min_silence_seconds=self.turn_gap_seconds,
        )

        return BaselineResult(
            name=self.info.mode.value,
            description=self.info.description,
            frame_seconds=self.internal_pause_seconds,
            min_silence_seconds=self.turn_gap_seconds,
            threshold=self.turn_gap_seconds,
            speech_segments=speech_segments,
            end_of_turn_events=end_of_turn_events,
        )


def transcript_gap_detector() -> TranscriptGapDetector:
    try:
        mode = EndOfTurnDetectorMode("transcript_gap")
    except ValueError as error:
        raise ValueError(
            "EndOfTurnDetectorMode.TRANSCRIPT_GAP must be added with value "
            "`transcript_gap` before wiring transcript_gap_detector()."
        ) from error

    return TranscriptGapDetector(
        info=EndOfTurnDetectorInfo(
            mode=mode,
            label="Transcript gap",
            description=(
                "Transcript-only speaker 1 turn spans split at 1.5 s word gaps; "
                "end-of-turn events fire after matching transcript gaps or trailing silence "
                "inside the 180 s analysis cap."
            ),
        ),
        turn_gap_seconds=TURN_GAP_SECONDS,
        internal_pause_seconds=INTERNAL_PAUSE_SECONDS,
    )


def _speaker_metadata_location(wave_path: Path) -> SpeakerMetadataLocation:
    suffix = "_speaker1"
    if not wave_path.name.endswith(f"{suffix}.wav"):
        raise ValueError(f"Expected a speaker 1 WAV path ending in `{suffix}.wav`: {wave_path}")

    session_identifier = wave_path.stem[: -len(suffix)]
    if not session_identifier:
        raise ValueError(f"Could not derive session identifier from WAV path: {wave_path}")

    return SpeakerMetadataLocation(
        metadata_path=wave_path.parent / f"{session_identifier}.json",
        speaker_name="Speaker1",
    )


def _read_session_transcript(metadata_path: Path) -> SessionTranscript:
    if not metadata_path.exists():
        raise ValueError(f"Missing transcript metadata beside WAV: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata_payload = cast(JsonValue, json.load(metadata_file))

    return _parse_session_transcript(metadata_payload=metadata_payload, metadata_path=metadata_path)


def _parse_session_transcript(
    metadata_payload: JsonValue,
    metadata_path: Path,
) -> SessionTranscript:
    metadata_object = _required_object(
        value=metadata_payload,
        field_name=f"metadata root in {metadata_path}",
    )
    speaker_transcripts_value = _required_field(
        source=metadata_object,
        field_name="speakerTranscript",
    )
    speaker_transcripts_array = _required_array(
        value=speaker_transcripts_value,
        field_name="speakerTranscript",
    )
    if not speaker_transcripts_array:
        raise ValueError(f"Missing speakerTranscript entries in metadata: {metadata_path}")

    return SessionTranscript(
        name=_required_string(
            value=_required_field(source=metadata_object, field_name="name"),
            field_name="name",
        ),
        duration_seconds=_required_float(
            value=_required_field(source=metadata_object, field_name="durationSeconds"),
            field_name="durationSeconds",
        ),
        speaker_transcripts=[
            _parse_speaker_transcript(
                speaker_transcript_value=speaker_transcript_value,
                speaker_transcript_index=speaker_transcript_index,
            )
            for speaker_transcript_index, speaker_transcript_value in enumerate(
                speaker_transcripts_array
            )
        ],
    )


def _parse_speaker_transcript(
    speaker_transcript_value: JsonValue,
    speaker_transcript_index: int,
) -> SpeakerTranscript:
    field_prefix = f"speakerTranscript[{speaker_transcript_index}]"
    speaker_transcript_object = _required_object(
        value=speaker_transcript_value,
        field_name=field_prefix,
    )
    words_value = _required_field(
        source=speaker_transcript_object,
        field_name="words",
    )
    words_array = _required_array(
        value=words_value,
        field_name=f"{field_prefix}.words",
    )
    return SpeakerTranscript(
        speaker=_required_string(
            value=_required_field(source=speaker_transcript_object, field_name="speaker"),
            field_name=f"{field_prefix}.speaker",
        ),
        words=[
            _parse_transcript_word(
                word_value=word_value,
                word_index=word_index,
                field_prefix=field_prefix,
            )
            for word_index, word_value in enumerate(words_array)
        ],
    )


def _parse_transcript_word(
    word_value: JsonValue,
    word_index: int,
    field_prefix: str,
) -> TranscriptWord:
    word_field_prefix = f"{field_prefix}.words[{word_index}]"
    word_object = _required_object(value=word_value, field_name=word_field_prefix)
    start_seconds = _required_float(
        value=_required_field(source=word_object, field_name="startTime"),
        field_name=f"{word_field_prefix}.startTime",
    )
    end_seconds = _required_float(
        value=_required_field(source=word_object, field_name="endTime"),
        field_name=f"{word_field_prefix}.endTime",
    )
    if end_seconds < start_seconds:
        raise ValueError(f"Transcript word endTime precedes startTime at {word_field_prefix}.")

    return TranscriptWord(
        text=_required_string(
            value=_required_field(source=word_object, field_name="text"),
            field_name=f"{word_field_prefix}.text",
        ),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _speaker_words(
    session_transcript: SessionTranscript,
    speaker_name: str,
    analysis_end_seconds: float,
) -> list[TranscriptWord]:
    selected_transcripts = [
        speaker_transcript
        for speaker_transcript in session_transcript.speaker_transcripts
        if speaker_transcript.speaker == speaker_name
    ]
    if not selected_transcripts:
        raise ValueError(
            f"Missing speaker transcript for {speaker_name} in {session_transcript.name}."
        )

    words = [
        _clipped_word(word=word, analysis_end_seconds=analysis_end_seconds)
        for speaker_transcript in selected_transcripts
        for word in speaker_transcript.words
        if _word_overlaps_analysis_window(
            word=word,
            analysis_end_seconds=analysis_end_seconds,
        )
    ]
    clipped_words = [word for word in words if word is not None]
    if not clipped_words:
        raise ValueError(
            f"Speaker transcript for {speaker_name} in {session_transcript.name} "
            "has no timed words inside the analysis window."
        )
    return sorted(clipped_words, key=lambda word: (word.start_seconds, word.end_seconds))


def _word_overlaps_analysis_window(
    word: TranscriptWord,
    analysis_end_seconds: float,
) -> bool:
    return word.start_seconds < analysis_end_seconds and word.end_seconds > 0.0


def _clipped_word(
    word: TranscriptWord,
    analysis_end_seconds: float,
) -> TranscriptWord | None:
    start_seconds = max(0.0, word.start_seconds)
    end_seconds = min(analysis_end_seconds, word.end_seconds)
    if end_seconds <= start_seconds:
        return None
    return TranscriptWord(
        text=word.text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _turn_segments(
    words: list[TranscriptWord],
    turn_gap_seconds: float,
) -> list[SpeechSegment]:
    if not words:
        return []

    speech_segments: list[SpeechSegment] = []
    current_start_seconds = words[0].start_seconds
    previous_end_seconds = words[0].end_seconds

    for word in words[1:]:
        gap_seconds = word.start_seconds - previous_end_seconds
        if gap_seconds > turn_gap_seconds:
            speech_segments.append(
                SpeechSegment(
                    start_seconds=_rounded_seconds(current_start_seconds),
                    end_seconds=_rounded_seconds(previous_end_seconds),
                )
            )
            current_start_seconds = word.start_seconds
        previous_end_seconds = max(previous_end_seconds, word.end_seconds)

    speech_segments.append(
        SpeechSegment(
            start_seconds=_rounded_seconds(current_start_seconds),
            end_seconds=_rounded_seconds(previous_end_seconds),
        )
    )
    return speech_segments


def _end_of_turn_events(
    speech_segments: list[SpeechSegment],
    analysis_end_seconds: float,
    min_silence_seconds: float,
) -> list[EndOfTurnEvent]:
    end_of_turn_events: list[EndOfTurnEvent] = []
    for segment_index, speech_segment in enumerate(speech_segments):
        next_segment_start_seconds = _next_segment_start_seconds(
            speech_segments=speech_segments,
            segment_index=segment_index,
            analysis_end_seconds=analysis_end_seconds,
        )
        silence_seconds = next_segment_start_seconds - speech_segment.end_seconds
        if silence_seconds < min_silence_seconds:
            continue

        end_of_turn_events.append(
            EndOfTurnEvent(
                time_seconds=speech_segment.end_seconds,
                speech_start_seconds=speech_segment.start_seconds,
                speech_end_seconds=speech_segment.end_seconds,
                silence_seconds=_rounded_seconds(silence_seconds),
            )
        )
    return end_of_turn_events


def _next_segment_start_seconds(
    speech_segments: list[SpeechSegment],
    segment_index: int,
    analysis_end_seconds: float,
) -> float:
    next_segment_index = segment_index + 1
    if next_segment_index < len(speech_segments):
        return speech_segments[next_segment_index].start_seconds
    return analysis_end_seconds


def _required_field(source: dict[str, JsonValue], field_name: str) -> JsonValue:
    if field_name not in source:
        raise ValueError(f"Missing required metadata field `{field_name}`.")
    return source[field_name]


def _required_object(value: JsonValue, field_name: str) -> dict[str, JsonValue]:
    match value:
        case dict():
            return value
        case _:
            raise ValueError(f"Expected `{field_name}` to be an object.")


def _required_array(value: JsonValue, field_name: str) -> list[JsonValue]:
    match value:
        case list():
            return value
        case _:
            raise ValueError(f"Expected `{field_name}` to be an array.")


def _required_string(value: JsonValue, field_name: str) -> str:
    match value:
        case str():
            return value
        case _:
            raise ValueError(f"Expected `{field_name}` to be a string.")


def _required_float(value: JsonValue, field_name: str) -> float:
    match value:
        case int() | float():
            return float(value)
        case _:
            raise ValueError(f"Expected `{field_name}` to be a number.")


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, ROUND_SECONDS_DIGITS)
