from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from app.local.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.local.analyses.end_of_turn.detectors.naive_vad import run_naive_vad_floor
from app.local.analyses.end_of_turn.service import (
    BackchannelSpan,
    BaselineResult,
    EndOfTurnEvent,
    InterruptionEvent,
    PauseSpan,
    SpeechSegment,
)
from app.shared.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

TARGET_SPEAKER = "Speaker1"
OTHER_SPEAKER = "Speaker2"
TURN_GAP_SECONDS = 1.5
INTERNAL_PAUSE_SECONDS = 0.5
BACKCHANNEL_MAX_SECONDS = 1.2
BACKCHANNEL_MAX_WORDS = 2
VAD_FRAME_SECONDS = 0.03
VAD_MIN_SPEECH_SECONDS = 0.12
VAD_MIN_SILENCE_SECONDS = 0.4
INTERRUPTION_PRE_END_MIN_SECONDS = 0.1
ROUND_SECONDS_DIGITS = 6
BACKCHANNEL_TOKENS = {
    "yeah",
    "yep",
    "yup",
    "yes",
    "ok",
    "okay",
    "right",
    "sure",
    "mhm",
    "mm",
    "hmm",
    "huh",
    "haha",
    "hah",
    "laugh",
    "laughs",
    "laughter",
    "chuckle",
    "chuckles",
}
BACKCHANNEL_PHRASES = {
    "uh huh",
    "uh-huh",
    "mm hmm",
    "mm-hmm",
    "mm hm",
    "mm-hm",
    "mhm mhm",
    "yeah yeah",
    "yep yep",
    "ok ok",
    "okay okay",
    "ha ha",
    "haha haha",
}


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TranscriptTurn:
    speaker: str
    text: str
    start_seconds: float
    end_seconds: float
    words: list[TranscriptWord]


@dataclass(frozen=True)
class SpeakerMetadataLocation:
    metadata_path: Path
    speaker_name: str


@dataclass(frozen=True)
class ClassifiedTurns:
    speech_segments: list[SpeechSegment]
    pause_spans: list[PauseSpan]
    backchannel_spans: list[BackchannelSpan]
    interruption_events: list[InterruptionEvent]


@dataclass(frozen=True)
class TwoSpeakerAnnotationDetector:
    info: EndOfTurnDetectorInfo
    turn_gap_seconds: float
    internal_pause_seconds: float

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        metadata_location = _speaker_metadata_location(wave_path=speaker1_path)
        turns = _read_transcript_turns(metadata_path=metadata_location.metadata_path)
        analysis_end_seconds = _analysis_end_seconds(turns=turns)
        return analyze_two_speaker_turns(
            speaker1_path=speaker1_path,
            turns=turns,
            analysis_end_seconds=analysis_end_seconds,
            result_name=self.info.mode.value,
            description=self.info.description,
            turn_gap_seconds=self.turn_gap_seconds,
            internal_pause_seconds=self.internal_pause_seconds,
        )


def two_speaker_annotation_detector() -> TwoSpeakerAnnotationDetector:
    return TwoSpeakerAnnotationDetector(
        info=EndOfTurnDetectorInfo(
            mode=EndOfTurnDetectorMode.TWO_SPEAKER_ANNOTATION,
            label="Two speaker annotation",
            description=(
                "Speaker 1 labels from both LUEL speakers plus naive VAD-only "
                "backchannels outside Speaker 1 turns; ambiguous short transcript turns "
                "remain speech."
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
        speaker_name=TARGET_SPEAKER,
    )


def _read_transcript_turns(metadata_path: Path) -> list[TranscriptTurn]:
    if not metadata_path.exists():
        raise ValueError(f"Missing transcript metadata beside WAV: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata_payload = cast(JsonValue, json.load(metadata_file))

    metadata_object = _required_object(
        value=metadata_payload,
        field_name=f"metadata root in {metadata_path}",
    )
    speaker_transcripts = _required_array(
        value=_required_field(source=metadata_object, field_name="speakerTranscript"),
        field_name="speakerTranscript",
    )
    return [
        _parse_transcript_turn(turn_value=turn_value, turn_index=turn_index)
        for turn_index, turn_value in enumerate(speaker_transcripts)
    ]


def _parse_transcript_turn(turn_value: JsonValue, turn_index: int) -> TranscriptTurn:
    field_prefix = f"speakerTranscript[{turn_index}]"
    turn_object = _required_object(value=turn_value, field_name=field_prefix)
    start_seconds = _required_float(
        value=_required_field(source=turn_object, field_name="startTime"),
        field_name=f"{field_prefix}.startTime",
    )
    end_seconds = _required_float(
        value=_required_field(source=turn_object, field_name="endTime"),
        field_name=f"{field_prefix}.endTime",
    )
    if end_seconds < start_seconds:
        raise ValueError(f"Transcript turn endTime precedes startTime at {field_prefix}.")

    words = _required_array(
        value=_required_field(source=turn_object, field_name="words"),
        field_name=f"{field_prefix}.words",
    )
    return TranscriptTurn(
        speaker=_required_string(
            value=_required_field(source=turn_object, field_name="speaker"),
            field_name=f"{field_prefix}.speaker",
        ),
        text=_required_string(
            value=_required_field(source=turn_object, field_name="text"),
            field_name=f"{field_prefix}.text",
        ),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        words=[
            _parse_transcript_word(
                word_value=word_value,
                word_index=word_index,
                field_prefix=field_prefix,
            )
            for word_index, word_value in enumerate(words)
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


def _analysis_end_seconds(turns: list[TranscriptTurn]) -> float:
    if not turns:
        raise ValueError("Cannot analyze two-speaker annotations without transcript turns.")
    transcript_end_seconds = max(turn.end_seconds for turn in turns)
    return min(transcript_end_seconds, ANALYSIS_AUDIO_MAX_DURATION_SECONDS)


def _clipped_turns(
    turns: list[TranscriptTurn],
    analysis_end_seconds: float,
) -> list[TranscriptTurn]:
    clipped_turns: list[TranscriptTurn] = []
    for turn in turns:
        if turn.start_seconds >= analysis_end_seconds or turn.end_seconds <= 0.0:
            continue
        clipped_words = [
            clipped_word
            for word in turn.words
            if word.start_seconds < analysis_end_seconds and word.end_seconds > 0.0
            for clipped_word in [
                _clipped_word(word=word, analysis_end_seconds=analysis_end_seconds)
            ]
            if clipped_word is not None
        ]
        start_seconds = max(0.0, turn.start_seconds)
        end_seconds = min(analysis_end_seconds, turn.end_seconds)
        if end_seconds <= start_seconds:
            continue
        clipped_turns.append(
            TranscriptTurn(
                speaker=turn.speaker,
                text=turn.text,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                words=clipped_words,
            )
        )
    return clipped_turns


def _clipped_word(
    word: TranscriptWord,
    analysis_end_seconds: float,
) -> TranscriptWord | None:
    start_seconds = max(0.0, word.start_seconds)
    end_seconds = min(analysis_end_seconds, word.end_seconds)
    if end_seconds <= start_seconds:
        return None
    return TranscriptWord(text=word.text, start_seconds=start_seconds, end_seconds=end_seconds)


def _classified_turns(
    speaker1_path: Path,
    turns: list[TranscriptTurn],
    turn_gap_seconds: float,
    internal_pause_seconds: float,
    target_speaker: str,
    other_speaker: str,
    include_vad_backchannels: bool,
) -> ClassifiedTurns:
    speech_segments: list[SpeechSegment] = []
    pause_spans: list[PauseSpan] = []
    transcript_backchannels: list[BackchannelSpan] = []
    interruption_events: list[InterruptionEvent] = []

    for turn_index, turn in enumerate(turns):
        if turn.speaker != target_speaker:
            continue
        if _is_contextual_backchannel(
            turns=turns,
            turn_index=turn_index,
            target_speaker=target_speaker,
            other_speaker=other_speaker,
        ):
            transcript_backchannels.append(_backchannel_span(turn=turn, text=turn.text))
            continue
        speech_segments.append(_speech_segment(turn=turn))
        pause_spans.extend(
            _pause_spans(
                turn=turn,
                internal_pause_seconds=internal_pause_seconds,
            )
        )
        interruption_event = _interruption_event(
            turns=turns,
            turn_index=turn_index,
            other_speaker=other_speaker,
        )
        if interruption_event is not None:
            interruption_events.append(interruption_event)

    vad_backchannels = (
        _vad_backchannel_spans(
            speaker1_path=speaker1_path,
            target_turns=[turn for turn in turns if turn.speaker == target_speaker],
        )
        if include_vad_backchannels
        else []
    )
    backchannel_spans = sorted(
        transcript_backchannels + vad_backchannels,
        key=lambda span: (span.start_seconds, span.end_seconds),
    )
    merged_speech_segments = _merge_speech_segments(
        speech_segments=speech_segments,
        turn_gap_seconds=turn_gap_seconds,
    )
    return ClassifiedTurns(
        speech_segments=merged_speech_segments,
        pause_spans=pause_spans,
        backchannel_spans=backchannel_spans,
        interruption_events=interruption_events,
    )


def _is_contextual_backchannel(
    turns: list[TranscriptTurn],
    turn_index: int,
    target_speaker: str,
    other_speaker: str,
) -> bool:
    turn = turns[turn_index]
    if not _is_short_backchannel_text(turn=turn):
        return False
    if _answers_previous_question(
        turns=turns,
        turn_index=turn_index,
        other_speaker=other_speaker,
    ):
        return False
    if _target_speaker_has_adjacent_content(
        turns=turns,
        turn_index=turn_index,
        target_speaker=target_speaker,
        other_speaker=other_speaker,
    ):
        return False
    return _other_speaker_continues(
        turns=turns,
        turn_index=turn_index,
        other_speaker=other_speaker,
    )


def _is_short_backchannel_text(turn: TranscriptTurn) -> bool:
    if len(turn.words) > BACKCHANNEL_MAX_WORDS:
        return False
    if turn.end_seconds - turn.start_seconds > BACKCHANNEL_MAX_SECONDS:
        return False

    normalized_tokens = [
        normalized_token
        for normalized_token in (_normalized_backchannel_token(word.text) for word in turn.words)
        if normalized_token
    ]
    if not normalized_tokens:
        return False

    phrase = " ".join(normalized_tokens)
    if phrase in BACKCHANNEL_PHRASES:
        return True
    if len(normalized_tokens) == 1:
        return normalized_tokens[0] in BACKCHANNEL_TOKENS
    return all(_is_laugh_token(token=token) for token in normalized_tokens)


def _answers_previous_question(
    turns: list[TranscriptTurn],
    turn_index: int,
    other_speaker: str,
) -> bool:
    previous_turn = _previous_speaker_turn(
        turns=turns,
        turn_index=turn_index,
        speaker=other_speaker,
    )
    if previous_turn is None:
        return False
    return previous_turn.text.strip().endswith("?")


def _target_speaker_has_adjacent_content(
    turns: list[TranscriptTurn],
    turn_index: int,
    target_speaker: str,
    other_speaker: str,
) -> bool:
    current_turn = turns[turn_index]
    for earlier_turn in reversed(turns[:turn_index]):
        if earlier_turn.speaker == other_speaker:
            break
        if earlier_turn.speaker == target_speaker:
            return current_turn.start_seconds - earlier_turn.end_seconds <= TURN_GAP_SECONDS

    for later_turn in turns[turn_index + 1 :]:
        if later_turn.speaker == other_speaker:
            return False
        if later_turn.speaker == target_speaker:
            return later_turn.start_seconds - current_turn.end_seconds <= TURN_GAP_SECONDS
    return False


def _other_speaker_continues(
    turns: list[TranscriptTurn],
    turn_index: int,
    other_speaker: str,
) -> bool:
    current_turn = turns[turn_index]
    previous_turn = _previous_speaker_turn(
        turns=turns,
        turn_index=turn_index,
        speaker=other_speaker,
    )
    next_turn = _next_speaker_turn(
        turns=turns,
        turn_index=turn_index,
        speaker=other_speaker,
    )
    previous_is_near = (
        previous_turn is not None
        and current_turn.start_seconds - previous_turn.end_seconds <= INTERNAL_PAUSE_SECONDS
    )
    next_is_near = (
        next_turn is not None
        and next_turn.start_seconds - current_turn.end_seconds <= TURN_GAP_SECONDS
    )
    return previous_is_near or next_is_near


def _previous_speaker_turn(
    turns: list[TranscriptTurn],
    turn_index: int,
    speaker: str,
) -> TranscriptTurn | None:
    for previous_turn in reversed(turns[:turn_index]):
        if previous_turn.speaker == speaker:
            return previous_turn
    return None


def _next_speaker_turn(
    turns: list[TranscriptTurn],
    turn_index: int,
    speaker: str,
) -> TranscriptTurn | None:
    for next_turn in turns[turn_index + 1 :]:
        if next_turn.speaker == speaker:
            return next_turn
    return None


def _speech_segment(turn: TranscriptTurn) -> SpeechSegment:
    return SpeechSegment(
        start_seconds=_rounded_seconds(turn.start_seconds),
        end_seconds=_rounded_seconds(turn.end_seconds),
    )


def _pause_spans(
    turn: TranscriptTurn,
    internal_pause_seconds: float,
) -> list[PauseSpan]:
    if not turn.words:
        return []

    pause_spans: list[PauseSpan] = []
    previous_word = turn.words[0]
    for word in turn.words[1:]:
        gap_seconds = word.start_seconds - previous_word.end_seconds
        if gap_seconds >= internal_pause_seconds:
            pause_spans.append(
                PauseSpan(
                    start_seconds=_rounded_seconds(previous_word.end_seconds),
                    end_seconds=_rounded_seconds(word.start_seconds),
                    duration_seconds=_rounded_seconds(gap_seconds),
                )
            )
        if word.end_seconds > previous_word.end_seconds:
            previous_word = word
    return pause_spans


def _merge_speech_segments(
    speech_segments: list[SpeechSegment],
    turn_gap_seconds: float,
) -> list[SpeechSegment]:
    if not speech_segments:
        return []

    sorted_segments = sorted(
        speech_segments,
        key=lambda segment: (segment.start_seconds, segment.end_seconds),
    )
    merged_segments = [sorted_segments[0]]
    for speech_segment in sorted_segments[1:]:
        previous_segment = merged_segments[-1]
        if speech_segment.start_seconds - previous_segment.end_seconds <= turn_gap_seconds:
            merged_segments[-1] = SpeechSegment(
                start_seconds=previous_segment.start_seconds,
                end_seconds=max(previous_segment.end_seconds, speech_segment.end_seconds),
            )
            continue
        merged_segments.append(speech_segment)
    return merged_segments


def _backchannel_span(turn: TranscriptTurn, text: str) -> BackchannelSpan:
    return BackchannelSpan(
        start_seconds=_rounded_seconds(turn.start_seconds),
        end_seconds=_rounded_seconds(turn.end_seconds),
        duration_seconds=_rounded_seconds(turn.end_seconds - turn.start_seconds),
        text=text,
    )


def _vad_backchannel_spans(
    speaker1_path: Path,
    target_turns: list[TranscriptTurn],
) -> list[BackchannelSpan]:
    vad_result = run_naive_vad_floor(
        wave_path=speaker1_path,
        result_name="speaker1_vad_for_annotations",
        description="Speaker 1 naive VAD for two-speaker annotation backchannels.",
        frame_seconds=VAD_FRAME_SECONDS,
        min_speech_seconds=VAD_MIN_SPEECH_SECONDS,
        min_silence_seconds=VAD_MIN_SILENCE_SECONDS,
    )
    vad_backchannels: list[BackchannelSpan] = []
    for speech_segment in vad_result.speech_segments:
        duration_seconds = speech_segment.end_seconds - speech_segment.start_seconds
        if duration_seconds > BACKCHANNEL_MAX_SECONDS:
            continue
        if _segment_overlaps_turns(speech_segment=speech_segment, turns=target_turns):
            continue
        vad_backchannels.append(
            BackchannelSpan(
                start_seconds=speech_segment.start_seconds,
                end_seconds=speech_segment.end_seconds,
                duration_seconds=_rounded_seconds(duration_seconds),
                text="[vad]",
            )
        )
    return vad_backchannels


def _segment_overlaps_turns(
    speech_segment: SpeechSegment,
    turns: list[TranscriptTurn],
) -> bool:
    return any(
        speech_segment.start_seconds < turn.end_seconds
        and speech_segment.end_seconds > turn.start_seconds
        for turn in turns
    )


def _interruption_event(
    turns: list[TranscriptTurn],
    turn_index: int,
    other_speaker: str,
) -> InterruptionEvent | None:
    turn = turns[turn_index]
    previous_turn = _previous_speaker_turn(
        turns=turns,
        turn_index=turn_index,
        speaker=other_speaker,
    )
    if previous_turn is None:
        return None

    start_before_other_end_seconds = previous_turn.end_seconds - turn.start_seconds
    if start_before_other_end_seconds < INTERRUPTION_PRE_END_MIN_SECONDS:
        return None

    return InterruptionEvent(
        time_seconds=_rounded_seconds(turn.start_seconds),
        confidence=1.0,
        interrupted_speaker=previous_turn.speaker,
        interrupting_speaker=turn.speaker,
        text=turn.text,
    )


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


def _normalized_backchannel_token(text: str) -> str:
    lower_text = text.strip().lower()
    bracket_stripped_text = lower_text.strip("[](){}")
    if _is_laugh_token(token=bracket_stripped_text):
        return bracket_stripped_text
    return re.sub(r"[^a-z0-9-]+", "", bracket_stripped_text)


def _is_laugh_token(token: str) -> bool:
    return token in {
        "ha",
        "haha",
        "hahaha",
        "hah",
        "laugh",
        "laughs",
        "laughter",
        "chuckle",
        "chuckles",
    }


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


def analyze_two_speaker_turns(
    speaker1_path: Path,
    turns: list[TranscriptTurn],
    analysis_end_seconds: float,
    result_name: str,
    description: str,
    turn_gap_seconds: float,
    internal_pause_seconds: float,
    target_speaker: str = TARGET_SPEAKER,
    other_speaker: str = OTHER_SPEAKER,
    include_vad_backchannels: bool = True,
) -> BaselineResult:
    clipped_turns = _clipped_turns(turns=turns, analysis_end_seconds=analysis_end_seconds)
    if not clipped_turns:
        raise ValueError("No timed two-speaker transcript turns exist inside the analysis window.")

    classified_turns = _classified_turns(
        speaker1_path=speaker1_path,
        turns=clipped_turns,
        turn_gap_seconds=turn_gap_seconds,
        internal_pause_seconds=internal_pause_seconds,
        target_speaker=target_speaker,
        other_speaker=other_speaker,
        include_vad_backchannels=include_vad_backchannels,
    )
    end_of_turn_events = _end_of_turn_events(
        speech_segments=classified_turns.speech_segments,
        analysis_end_seconds=analysis_end_seconds,
        min_silence_seconds=turn_gap_seconds,
    )
    return BaselineResult(
        name=result_name,
        description=description,
        frame_seconds=internal_pause_seconds,
        min_silence_seconds=turn_gap_seconds,
        threshold=turn_gap_seconds,
        speech_segments=classified_turns.speech_segments,
        pause_spans=classified_turns.pause_spans,
        backchannel_spans=classified_turns.backchannel_spans,
        end_of_turn_events=end_of_turn_events,
        interruption_events=classified_turns.interruption_events,
    )


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, ROUND_SECONDS_DIGITS)
