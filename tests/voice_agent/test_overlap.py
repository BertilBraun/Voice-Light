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


def test_active_speech_takes_the_floor_at_the_hard_deadline() -> None:
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
    assert decision.kind is OverlapResolutionKind.FLOOR_TAKING
