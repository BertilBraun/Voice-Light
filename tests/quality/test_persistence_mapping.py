from __future__ import annotations

from uuid import uuid4

from app.ingestion.service import quality_result_input
from app.quality.models import ProcessingStatus, QualityResult


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
        event_candidates=(),
        raw_quality_score=None,
        calibrated_quality_score=None,
        calibration_flags=("bad_audio",),
        total_quality_score=None,
        error="ValueError: bad audio",
    )

    record = quality_result_input(sample_id, quality_result)

    assert record.sample_id == sample_id
    assert record.metric_version == "quality-test"
    assert record.status == "failed"
    assert record.flags == ("bad_audio",)
    assert record.payload["error"] == "ValueError: bad audio"
