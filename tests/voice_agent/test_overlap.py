from __future__ import annotations

import pytest

from app.compute.voice.overlap import (
    OverlapEvidence,
    OverlapResolutionKind,
    ProvisionalOverlapPolicyConfig,
    ProvisionalVadTranscriptOverlapPolicy,
)


@pytest.mark.parametrize("transcript", ["", "mm-hm", "uh-huh", "okay", "right", "yeah"])
def test_short_finished_feedback_does_not_take_the_floor(transcript: str) -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=300,
            speech_active=False,
            transcript=transcript,
            transcript_event_id="transcript-1" if transcript else None,
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )
    assert decision.kind is OverlapResolutionKind.NON_FLOOR_TAKING


@pytest.mark.parametrize("transcript", ["ha", "haha", "laughter"])
def test_short_laughter_does_not_take_the_floor(transcript: str) -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=250,
            speech_active=False,
            transcript=transcript,
            transcript_event_id="transcript-1",
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )
    assert decision.kind is OverlapResolutionKind.NON_FLOOR_TAKING


@pytest.mark.parametrize("transcript", ["how?", "what?", "no", "no, wait", "actually"])
def test_question_and_repair_language_requires_a_response(transcript: str) -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=150,
            speech_active=True,
            transcript=transcript,
            transcript_event_id="transcript-1",
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )
    assert decision.kind is OverlapResolutionKind.RESPONSE_REQUIRED
    assert decision.fast_path


@pytest.mark.parametrize("transcript", ["yeah but", "yeah, but this is wrong", "right and then"])
def test_acknowledgement_prefix_with_continuation_takes_the_floor(transcript: str) -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=200,
            speech_active=True,
            transcript=transcript,
            transcript_event_id="transcript-1",
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )
    assert decision.kind is OverlapResolutionKind.FLOOR_TAKING


def test_transcript_free_active_speech_remains_reversible_at_classification_deadline() -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=500,
            speech_active=True,
            transcript="",
            transcript_event_id=None,
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )
    assert decision.kind is OverlapResolutionKind.UNRESOLVED


@pytest.mark.parametrize("transcript", ["A", "Y", "O", "I want to add something"])
def test_unfinished_lexical_evidence_remains_reversible_while_speech_is_active(
    transcript: str,
) -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=500,
            speech_active=True,
            transcript=transcript,
            transcript_event_id="transcript-1",
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )

    assert decision.kind is OverlapResolutionKind.UNRESOLVED


def test_finished_lexical_evidence_requires_a_response() -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=500,
            speech_active=False,
            transcript="I want to add something",
            transcript_event_id="transcript-1",
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )

    assert decision.kind is OverlapResolutionKind.RESPONSE_REQUIRED


@pytest.mark.parametrize("transcript", ["mm-hm", "uh-huh", "yeah", "yes", "haha"])
def test_active_feedback_remains_reversible_at_classification_deadline(
    transcript: str,
) -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=500,
            speech_active=True,
            transcript=transcript,
            transcript_event_id="transcript-1",
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )
    assert decision.kind is OverlapResolutionKind.UNRESOLVED


def test_transcript_free_active_speech_takes_the_floor_at_hard_deadline() -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=900,
            speech_active=True,
            transcript="",
            transcript_event_id=None,
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )
    assert decision.kind is OverlapResolutionKind.FLOOR_TAKING


def test_short_transcript_free_backchannel_remains_reversible_before_hard_deadline() -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=800,
            speech_active=True,
            transcript="",
            transcript_event_id=None,
            interruption_probability=None,
            interruption_evidence_event_id=None,
        )
    )

    assert decision.kind is OverlapResolutionKind.UNRESOLVED


def test_strong_non_floor_feedback_prediction_waits_for_speech_end() -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())

    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=160,
            speech_active=True,
            transcript="",
            transcript_event_id=None,
            interruption_probability=0.1,
            interruption_evidence_event_id="prediction-1",
            non_floor_feedback_probability=0.9,
            non_floor_feedback_evidence_event_id="prediction-1",
        )
    )

    assert decision.kind is OverlapResolutionKind.UNRESOLVED


def test_strong_non_floor_feedback_prediction_resumes_at_speech_end() -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())
    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=240,
            speech_active=False,
            transcript="",
            transcript_event_id=None,
            interruption_probability=0.1,
            interruption_evidence_event_id="prediction-1",
            non_floor_feedback_probability=0.9,
            non_floor_feedback_evidence_event_id="prediction-1",
        )
    )
    assert decision.kind is OverlapResolutionKind.NON_FLOOR_TAKING
    assert decision.fast_path


def test_strong_floor_take_prediction_commits_interruption_before_deadline() -> None:
    policy = ProvisionalVadTranscriptOverlapPolicy(ProvisionalOverlapPolicyConfig())

    decision = policy.classify(
        OverlapEvidence(
            elapsed_ms=160,
            speech_active=True,
            transcript="",
            transcript_event_id=None,
            interruption_probability=0.83,
            interruption_evidence_event_id="prediction-1",
            non_floor_feedback_probability=0.1,
            non_floor_feedback_evidence_event_id="prediction-1",
        )
    )

    assert decision.kind is OverlapResolutionKind.FLOOR_TAKING
    assert decision.fast_path
