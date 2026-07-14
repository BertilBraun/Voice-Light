from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.local.analyses.end_of_turn.detectors.two_speaker_annotation import (
    BACKCHANNEL_PHRASES,
    BACKCHANNEL_TOKENS,
    TranscriptTurn,
)
from app.local.analyses.end_of_turn.service import (
    BackchannelSpan,
    ConnectionHypothesis,
    EndOfTurnEvent,
    InterruptionEvent,
    PauseSpan,
    SegmentHypothesis,
    SpeechSegment,
)

ROUND_SECONDS_DIGITS = 6


@dataclass(frozen=True)
class ConversationScoringConfig:
    backchannel_threshold: float = 0.55
    turn_threshold: float = 0.5
    interruption_threshold: float = 0.55
    merge_threshold: float = 0.5
    pause_threshold: float = 0.2
    end_of_turn_silence_seconds: float = 1.5


@dataclass(frozen=True)
class ScoredConversation:
    speech_segments: list[SpeechSegment]
    pause_spans: list[PauseSpan]
    backchannel_spans: list[BackchannelSpan]
    interruption_events: list[InterruptionEvent]
    end_of_turn_events: list[EndOfTurnEvent]
    segment_hypotheses: list[SegmentHypothesis]
    connection_hypotheses: list[ConnectionHypothesis]


def score_conversation(
    turns: list[TranscriptTurn],
    target_speaker: str,
    other_speaker: str,
    analysis_end_seconds: float,
    config: ConversationScoringConfig,
) -> ScoredConversation:
    target_turns = sorted(
        (turn for turn in turns if turn.speaker == target_speaker),
        key=lambda turn: (turn.start_seconds, turn.end_seconds),
    )
    other_turns = sorted(
        (turn for turn in turns if turn.speaker == other_speaker),
        key=lambda turn: (turn.start_seconds, turn.end_seconds),
    )
    segment_hypotheses = [
        _segment_hypothesis(target_turn=turn, other_turns=other_turns) for turn in target_turns
    ]
    connection_hypotheses = [
        _connection_hypothesis(
            earlier_turn=earlier_turn,
            later_turn=later_turn,
            other_turns=other_turns,
        )
        for earlier_turn, later_turn in zip(target_turns, target_turns[1:], strict=False)
    ]
    speech_segments = [
        SpeechSegment(
            start_seconds=_rounded(turn.start_seconds),
            end_seconds=_rounded(turn.end_seconds),
        )
        for turn in target_turns
    ]
    backchannel_spans = [
        BackchannelSpan(
            start_seconds=_rounded(turn.start_seconds),
            end_seconds=_rounded(turn.end_seconds),
            duration_seconds=_rounded(turn.end_seconds - turn.start_seconds),
            text=turn.text,
        )
        for turn, hypothesis in zip(target_turns, segment_hypotheses, strict=True)
        if hypothesis.backchannel_confidence >= config.backchannel_threshold
    ]
    interruption_events = [
        InterruptionEvent(
            time_seconds=_rounded(turn.start_seconds),
            confidence=hypothesis.interruption_confidence,
            interrupted_speaker=other_speaker,
            interrupting_speaker=target_speaker,
            text=turn.text,
        )
        for turn, hypothesis in zip(target_turns, segment_hypotheses, strict=True)
        if hypothesis.interruption_confidence >= config.interruption_threshold
    ]
    pause_spans = [
        PauseSpan(
            start_seconds=connection.earlier_end_seconds,
            end_seconds=connection.later_start_seconds,
            duration_seconds=connection.gap_seconds,
        )
        for connection in connection_hypotheses
        if connection.pause_confidence >= config.pause_threshold
    ]
    logical_turns = _logical_turns(
        target_turns=target_turns,
        segment_hypotheses=segment_hypotheses,
        connection_hypotheses=connection_hypotheses,
        config=config,
    )
    return ScoredConversation(
        speech_segments=speech_segments,
        pause_spans=pause_spans,
        backchannel_spans=backchannel_spans,
        interruption_events=interruption_events,
        end_of_turn_events=_end_of_turn_events(
            speech_segments=logical_turns,
            analysis_end_seconds=analysis_end_seconds,
            min_silence_seconds=config.end_of_turn_silence_seconds,
        ),
        segment_hypotheses=segment_hypotheses,
        connection_hypotheses=connection_hypotheses,
    )


def _segment_hypothesis(
    target_turn: TranscriptTurn,
    other_turns: list[TranscriptTurn],
) -> SegmentHypothesis:
    duration_seconds = target_turn.end_seconds - target_turn.start_seconds
    duration_shortness = 1.0 - _smoothstep(0.6, 2.0, duration_seconds)
    word_shortness = _word_shortness(word_count=len(target_turn.words))
    lexical_confidence = _lexical_backchannel_confidence(turn=target_turn)
    other_speaker_continuity = _other_speaker_continuity(
        target_turn=target_turn,
        other_turns=other_turns,
    )
    backchannel_confidence = _bounded(
        other_speaker_continuity
        * (0.55 * duration_shortness + 0.3 * word_shortness + 0.15 * lexical_confidence)
    )
    turn_confidence = 1.0 - backchannel_confidence
    interruption_confidence = _bounded(
        _other_speaker_active_at_onset(
            target_turn=target_turn,
            other_turns=other_turns,
        )
        * turn_confidence
    )
    return SegmentHypothesis(
        start_seconds=_rounded(target_turn.start_seconds),
        end_seconds=_rounded(target_turn.end_seconds),
        text=target_turn.text,
        backchannel_confidence=_rounded(backchannel_confidence),
        turn_confidence=_rounded(turn_confidence),
        interruption_confidence=_rounded(interruption_confidence),
    )


def _connection_hypothesis(
    earlier_turn: TranscriptTurn,
    later_turn: TranscriptTurn,
    other_turns: list[TranscriptTurn],
) -> ConnectionHypothesis:
    gap_seconds = max(0.0, later_turn.start_seconds - earlier_turn.end_seconds)
    time_merge_confidence = 1.0 / (1.0 + math.exp((gap_seconds - 0.9) / 0.28))
    other_speaker_coverage = _interval_coverage(
        start_seconds=earlier_turn.end_seconds,
        end_seconds=later_turn.start_seconds,
        turns=other_turns,
    )
    merge_confidence = _bounded(time_merge_confidence * (1.0 - 0.85 * other_speaker_coverage))
    pause_presence = _smoothstep(0.12, 0.55, gap_seconds)
    pause_confidence = _bounded(pause_presence * merge_confidence)
    return ConnectionHypothesis(
        earlier_end_seconds=_rounded(earlier_turn.end_seconds),
        later_start_seconds=_rounded(later_turn.start_seconds),
        gap_seconds=_rounded(gap_seconds),
        pause_confidence=_rounded(pause_confidence),
        merge_confidence=_rounded(merge_confidence),
    )


def _logical_turns(
    target_turns: list[TranscriptTurn],
    segment_hypotheses: list[SegmentHypothesis],
    connection_hypotheses: list[ConnectionHypothesis],
    config: ConversationScoringConfig,
) -> list[SpeechSegment]:
    logical_turns: list[SpeechSegment] = []
    previous_target_index: int | None = None
    for target_index, (turn, hypothesis) in enumerate(
        zip(target_turns, segment_hypotheses, strict=True)
    ):
        if hypothesis.turn_confidence < config.turn_threshold:
            previous_target_index = None
            continue
        speech_segment = SpeechSegment(
            start_seconds=_rounded(turn.start_seconds),
            end_seconds=_rounded(turn.end_seconds),
        )
        should_merge = (
            previous_target_index is not None
            and previous_target_index == target_index - 1
            and connection_hypotheses[previous_target_index].merge_confidence
            >= config.merge_threshold
        )
        if should_merge:
            previous_segment = logical_turns[-1]
            logical_turns[-1] = SpeechSegment(
                start_seconds=previous_segment.start_seconds,
                end_seconds=speech_segment.end_seconds,
            )
        else:
            logical_turns.append(speech_segment)
        previous_target_index = target_index
    return logical_turns


def _end_of_turn_events(
    speech_segments: list[SpeechSegment],
    analysis_end_seconds: float,
    min_silence_seconds: float,
) -> list[EndOfTurnEvent]:
    events: list[EndOfTurnEvent] = []
    for segment_index, speech_segment in enumerate(speech_segments):
        next_start_seconds = (
            speech_segments[segment_index + 1].start_seconds
            if segment_index + 1 < len(speech_segments)
            else analysis_end_seconds
        )
        silence_seconds = next_start_seconds - speech_segment.end_seconds
        if silence_seconds < min_silence_seconds:
            continue
        events.append(
            EndOfTurnEvent(
                time_seconds=speech_segment.end_seconds,
                speech_start_seconds=speech_segment.start_seconds,
                speech_end_seconds=speech_segment.end_seconds,
                silence_seconds=_rounded(silence_seconds),
            )
        )
    return events


def _other_speaker_continuity(
    target_turn: TranscriptTurn,
    other_turns: list[TranscriptTurn],
) -> float:
    target_duration_seconds = target_turn.end_seconds - target_turn.start_seconds
    if target_duration_seconds <= 0.0:
        return 0.0
    continuity_scores: list[float] = []
    for other_turn in other_turns:
        overlap_seconds = _overlap_seconds(
            first_start_seconds=target_turn.start_seconds,
            first_end_seconds=target_turn.end_seconds,
            second_start_seconds=other_turn.start_seconds,
            second_end_seconds=other_turn.end_seconds,
        )
        if overlap_seconds <= 0.0:
            continue
        overlap_fraction = overlap_seconds / target_duration_seconds
        lead_confidence = _smoothstep(
            0.0,
            0.3,
            target_turn.start_seconds - other_turn.start_seconds,
        )
        trail_confidence = _smoothstep(
            0.0,
            0.3,
            other_turn.end_seconds - target_turn.end_seconds,
        )
        continuity_scores.append(
            overlap_fraction * (0.5 + 0.25 * lead_confidence + 0.25 * trail_confidence)
        )
    bridged_continuity = _bridged_other_speaker_continuity(
        target_turn=target_turn,
        other_turns=other_turns,
    )
    return max([*continuity_scores, bridged_continuity])


def _bridged_other_speaker_continuity(
    target_turn: TranscriptTurn,
    other_turns: list[TranscriptTurn],
) -> float:
    earlier_turns = [turn for turn in other_turns if turn.end_seconds <= target_turn.start_seconds]
    later_turns = [turn for turn in other_turns if turn.start_seconds >= target_turn.end_seconds]
    if not earlier_turns or not later_turns:
        return 0.0
    earlier_turn = max(earlier_turns, key=lambda turn: turn.end_seconds)
    later_turn = min(later_turns, key=lambda turn: turn.start_seconds)
    gap_seconds = later_turn.start_seconds - earlier_turn.end_seconds
    if gap_seconds <= 0.0:
        return 0.0
    return 1.0 / (1.0 + math.exp((gap_seconds - 1.8) / 0.3))


def _other_speaker_active_at_onset(
    target_turn: TranscriptTurn,
    other_turns: list[TranscriptTurn],
) -> float:
    onset_scores = [
        min(
            _smoothstep(0.05, 0.5, target_turn.start_seconds - other_turn.start_seconds),
            _smoothstep(0.05, 0.5, other_turn.end_seconds - target_turn.start_seconds),
        )
        for other_turn in other_turns
        if other_turn.start_seconds < target_turn.start_seconds < other_turn.end_seconds
    ]
    return max(onset_scores, default=0.0)


def _interval_coverage(
    start_seconds: float,
    end_seconds: float,
    turns: list[TranscriptTurn],
) -> float:
    duration_seconds = end_seconds - start_seconds
    if duration_seconds <= 0.0:
        return 0.0
    covered_seconds = sum(
        _overlap_seconds(
            first_start_seconds=start_seconds,
            first_end_seconds=end_seconds,
            second_start_seconds=turn.start_seconds,
            second_end_seconds=turn.end_seconds,
        )
        for turn in turns
    )
    return _bounded(covered_seconds / duration_seconds)


def _lexical_backchannel_confidence(turn: TranscriptTurn) -> float:
    normalized_tokens = [
        token for token in (_normalized_token(word.text) for word in turn.words) if token
    ]
    phrase = " ".join(normalized_tokens)
    if phrase in BACKCHANNEL_PHRASES:
        return 1.0
    if normalized_tokens and all(token in BACKCHANNEL_TOKENS for token in normalized_tokens):
        return 1.0
    return 0.15 if len(normalized_tokens) <= 2 else 0.0


def _word_shortness(word_count: int) -> float:
    if word_count <= 2:
        return 1.0
    if word_count == 3:
        return 0.65
    if word_count == 4:
        return 0.3
    return 0.0


def _normalized_token(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "", text.strip().lower().strip("[](){}"))


def _overlap_seconds(
    first_start_seconds: float,
    first_end_seconds: float,
    second_start_seconds: float,
    second_end_seconds: float,
) -> float:
    return max(
        0.0,
        min(first_end_seconds, second_end_seconds) - max(first_start_seconds, second_start_seconds),
    )


def _smoothstep(edge_start: float, edge_end: float, value: float) -> float:
    if edge_end <= edge_start:
        raise ValueError("Smoothstep end must be greater than start.")
    position = _bounded((value - edge_start) / (edge_end - edge_start))
    return position * position * (3.0 - 2.0 * position)


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, value))


def _rounded(value: float) -> float:
    return round(value, ROUND_SECONDS_DIGITS)
