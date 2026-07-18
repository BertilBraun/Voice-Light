from __future__ import annotations

from uuid import uuid4

from app.local.ingestion.service import QualityResultProvenance, quality_result_input
from app.shared.quality import ProcessingStatus, QualityResult


def test_quality_result_input_preserves_failed_payload() -> None:
    sample_id = uuid4()
    quality_result = QualityResult(
        metric_version="quality-test",
        sample_id="sample-a",
        status=ProcessingStatus.FAILED,
        speaker1_uri="speaker1.wav",
        speaker2_uri="speaker2.wav",
        duration_seconds=None,
        interaction_density=None,
        timing_reliability=None,
        audio_quality=None,
        conversation_annotation=None,
        conversation_count_estimate=None,
        event_candidates=(),
        raw_quality_score=None,
        quality_flags=("bad_audio",),
        total_quality_score=None,
        error="ValueError: bad audio",
    )

    record = quality_result_input(
        sample_id=sample_id,
        quality_result=quality_result,
        provenance=None,
    )

    assert record.sample_id == sample_id
    assert record.metric_version == "quality-test"
    assert record.status == "failed"
    assert record.flags == ("bad_audio",)
    assert record.speaker1_parakeet_full_asr_transcript_id is None
    assert record.speaker2_parakeet_full_asr_transcript_id is None
    assert record.speaker1_canary_full_asr_transcript_id is None
    assert record.speaker2_canary_full_asr_transcript_id is None
    assert record.payload["error"] == "ValueError: bad audio"


def test_quality_result_input_persists_full_asr_provenance() -> None:
    sample_id = uuid4()
    speaker1_parakeet_transcript_id = uuid4()
    speaker2_parakeet_transcript_id = uuid4()
    speaker1_canary_transcript_id = uuid4()
    speaker2_canary_transcript_id = uuid4()
    quality_result = QualityResult(
        metric_version="quality-conversation-full-parakeet-v5",
        sample_id="sample-a",
        status=ProcessingStatus.COMPLETED,
        speaker1_uri="speaker1.wav",
        speaker2_uri="speaker2.wav",
        duration_seconds=10.0,
        interaction_density=None,
        timing_reliability=None,
        audio_quality=None,
        conversation_annotation=None,
        conversation_count_estimate=None,
        event_candidates=(),
        raw_quality_score=0.5,
        quality_flags=(),
        total_quality_score=0.5,
        error=None,
    )

    record = quality_result_input(
        sample_id=sample_id,
        quality_result=quality_result,
        provenance=QualityResultProvenance(
            speaker1_parakeet_full_asr_transcript_id=speaker1_parakeet_transcript_id,
            speaker2_parakeet_full_asr_transcript_id=speaker2_parakeet_transcript_id,
            speaker1_canary_full_asr_transcript_id=speaker1_canary_transcript_id,
            speaker2_canary_full_asr_transcript_id=speaker2_canary_transcript_id,
        ),
    )

    assert record.speaker1_parakeet_full_asr_transcript_id == speaker1_parakeet_transcript_id
    assert record.speaker2_parakeet_full_asr_transcript_id == speaker2_parakeet_transcript_id
    assert record.speaker1_canary_full_asr_transcript_id == speaker1_canary_transcript_id
    assert record.speaker2_canary_full_asr_transcript_id == speaker2_canary_transcript_id
