from app.local.analyses.end_of_turn.conversation_scoring import (
    ConversationScoringConfig,
    score_conversation,
)
from app.local.analyses.end_of_turn.detectors.two_speaker_annotation import (
    TranscriptTurn,
    TranscriptWord,
)
from app.local.analyses.end_of_turn.service import SegmentEvidenceSource, SpeechSegment


def test_short_nonlexical_segment_inside_continuous_speech_scores_as_backchannel() -> None:
    target_turn = _turn(
        speaker="Speaker1",
        words=(
            ("strict", 0.8, 1.0),
            ("thought", 1.0, 1.2),
        ),
    )
    other_turn = _turn(
        speaker="Speaker2",
        words=(("continuous", 0.0, 3.0),),
    )

    scored = score_conversation(
        turns=[other_turn, target_turn],
        audio_activity_segments=[],
        target_speaker="Speaker1",
        other_speaker="Speaker2",
        analysis_end_seconds=4.0,
        config=ConversationScoringConfig(),
    )

    hypothesis = scored.segment_hypotheses[0]
    assert hypothesis.keep_playing_confidence > 0.8
    assert hypothesis.turn_confidence < 0.2
    assert hypothesis.interruption_confidence < 0.2
    assert len(scored.backchannel_spans) == 1
    assert scored.interruption_events == []


def test_long_contentful_segment_starting_during_other_speech_scores_as_interruption() -> None:
    target_turn = _turn(
        speaker="Speaker1",
        words=(
            ("this", 0.8, 1.0),
            ("is", 1.0, 1.2),
            ("a", 1.2, 1.4),
            ("new", 1.4, 1.8),
            ("turn", 1.8, 2.5),
        ),
    )
    other_turn = _turn(
        speaker="Speaker2",
        words=(("speaking", 0.0, 1.5),),
    )

    scored = score_conversation(
        turns=[other_turn, target_turn],
        audio_activity_segments=[],
        target_speaker="Speaker1",
        other_speaker="Speaker2",
        analysis_end_seconds=4.0,
        config=ConversationScoringConfig(),
    )

    hypothesis = scored.segment_hypotheses[0]
    assert hypothesis.keep_playing_confidence < 0.2
    assert hypothesis.turn_confidence > 0.8
    assert hypothesis.interruption_confidence > 0.8
    assert scored.backchannel_spans == []
    assert len(scored.interruption_events) == 1


def test_short_reply_during_other_speaker_gap_scores_as_turn() -> None:
    target_turn = _turn(
        speaker="Speaker1",
        words=(("unexpected", 1.0, 1.25),),
    )
    other_turns = [
        _turn(speaker="Speaker2", words=(("before", 0.0, 0.9),)),
        _turn(speaker="Speaker2", words=(("after", 1.35, 2.0),)),
    ]

    scored = score_conversation(
        turns=[*other_turns, target_turn],
        audio_activity_segments=[],
        target_speaker="Speaker1",
        other_speaker="Speaker2",
        analysis_end_seconds=3.0,
        config=ConversationScoringConfig(),
    )

    hypothesis = scored.segment_hypotheses[0]
    assert hypothesis.keep_playing_confidence == 0.0
    assert hypothesis.turn_confidence == 1.0
    assert hypothesis.interruption_confidence == 0.0


def test_pause_confidence_rises_until_one_point_five_seconds_then_falls() -> None:
    turns = [
        _turn(speaker="Speaker1", words=(("one", 0.0, 0.4),)),
        _turn(speaker="Speaker1", words=(("two", 0.55, 0.9),)),
        _turn(speaker="Speaker1", words=(("three", 2.1, 2.4),)),
        _turn(speaker="Speaker1", words=(("four", 4.4, 4.7),)),
        _turn(speaker="Speaker1", words=(("five", 7.3, 7.6),)),
    ]

    scored = score_conversation(
        turns=turns,
        audio_activity_segments=[],
        target_speaker="Speaker1",
        other_speaker="Speaker2",
        analysis_end_seconds=8.0,
        config=ConversationScoringConfig(),
    )

    short_gap, likely_pause, long_pause, separate_turn = scored.connection_hypotheses
    assert short_gap.merge_confidence == 1.0
    assert likely_pause.merge_confidence == 1.0
    assert long_pause.merge_confidence == 0.5
    assert separate_turn.merge_confidence == 0.0
    assert short_gap.pause_confidence < likely_pause.pause_confidence
    assert likely_pause.pause_confidence > long_pause.pause_confidence
    assert long_pause.pause_confidence > separate_turn.pause_confidence


def test_partner_backchannel_does_not_prevent_pause_merge() -> None:
    turns = [
        _turn(speaker="Speaker1", words=(("before", 0.0, 0.5),)),
        _turn(speaker="Speaker2", words=(("yeah", 0.8, 1.1),)),
        _turn(speaker="Speaker1", words=(("after", 1.5, 2.0),)),
    ]

    scored = score_conversation(
        turns=turns,
        audio_activity_segments=[],
        target_speaker="Speaker1",
        other_speaker="Speaker2",
        analysis_end_seconds=4.0,
        config=ConversationScoringConfig(),
    )

    connection = scored.connection_hypotheses[0]
    assert connection.pause_confidence == 1.0
    assert connection.merge_confidence == 1.0
    assert len(scored.end_of_turn_events) == 1
    assert scored.end_of_turn_events[0].speech_start_seconds == 0.0
    assert scored.end_of_turn_events[0].speech_end_seconds == 2.0


def test_substantive_partner_turn_prevents_pause_merge() -> None:
    turns = [
        _turn(speaker="Speaker1", words=(("before", 0.0, 0.5),)),
        _turn(
            speaker="Speaker2",
            words=(
                ("that", 0.6, 0.75),
                ("is", 0.75, 0.9),
                ("my", 0.9, 1.05),
                ("actual", 1.05, 1.2),
                ("answer", 1.2, 1.4),
            ),
        ),
        _turn(speaker="Speaker1", words=(("after", 1.5, 2.0),)),
    ]

    scored = score_conversation(
        turns=turns,
        audio_activity_segments=[],
        target_speaker="Speaker1",
        other_speaker="Speaker2",
        analysis_end_seconds=3.0,
        config=ConversationScoringConfig(),
    )

    connection = scored.connection_hypotheses[0]
    assert connection.pause_confidence < 0.5
    assert connection.merge_confidence < 0.5


def test_short_overlap_does_not_score_as_interruption() -> None:
    target_turn = _turn(
        speaker="Speaker1",
        words=(("reply", 1.0, 1.5),),
    )
    other_turn = _turn(
        speaker="Speaker2",
        words=(("question", 0.0, 1.1),),
    )

    scored = score_conversation(
        turns=[other_turn, target_turn],
        audio_activity_segments=[],
        target_speaker="Speaker1",
        other_speaker="Speaker2",
        analysis_end_seconds=3.0,
        config=ConversationScoringConfig(),
    )

    hypothesis = scored.segment_hypotheses[0]
    assert hypothesis.turn_confidence > 0.8
    assert hypothesis.interruption_confidence == 0.0


def test_untranscribed_audio_activity_is_exposed_as_keep_playing_candidate() -> None:
    scored = score_conversation(
        turns=[],
        audio_activity_segments=[SpeechSegment(start_seconds=1.0, end_seconds=1.4)],
        target_speaker="Speaker1",
        other_speaker="Speaker2",
        analysis_end_seconds=3.0,
        config=ConversationScoringConfig(),
    )

    hypothesis = scored.segment_hypotheses[0]
    assert hypothesis.evidence_source == SegmentEvidenceSource.AUDIO_ACTIVITY
    assert hypothesis.keep_playing_confidence == 0.85
    assert hypothesis.turn_confidence == 0.15
    assert hypothesis.interruption_confidence == 0.0
    assert scored.backchannel_spans[0].text == "[untranscribed audio activity]"


def _turn(
    speaker: str,
    words: tuple[tuple[str, float, float], ...],
) -> TranscriptTurn:
    transcript_words = [
        TranscriptWord(text=text, start_seconds=start_seconds, end_seconds=end_seconds)
        for text, start_seconds, end_seconds in words
    ]
    return TranscriptTurn(
        speaker=speaker,
        text=" ".join(word.text for word in transcript_words),
        start_seconds=transcript_words[0].start_seconds,
        end_seconds=transcript_words[-1].end_seconds,
        words=transcript_words,
    )
