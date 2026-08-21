from app.training.turn_taking.benchmark_models import AudioProvenanceRecord
from app.training.turn_taking.benchmark_overlap import audit_smart_turn_overlap


def test_overlap_audit_flags_exact_hash_and_mundo_provenance() -> None:
    shared_hash = "1" * 64
    report = audit_smart_turn_overlap(
        local_records=(
            AudioProvenanceRecord(
                source_name="dataset_3",
                external_id="local-1",
                audio_sha256=shared_hash,
                pcm_sha256=None,
            ),
        ),
        smart_turn_records=(
            AudioProvenanceRecord(
                source_name="MundoAI",
                external_id="smart-turn-1",
                audio_sha256=shared_hash,
                pcm_sha256=None,
            ),
        ),
        external_repository="pipecat-ai/smart-turn-data-v3.2-train",
        external_revision="2" * 40,
    )

    assert report.provenance_risk_sources == ("dataset3",)
    assert report.exact_overlaps[0].matched_hash_kind == "audio_sha256"
    assert not report.clean_comparative_claim_permitted


def test_overlap_audit_requires_hashes_for_clean_claim() -> None:
    report = audit_smart_turn_overlap(
        local_records=(
            AudioProvenanceRecord(
                source_name="independent",
                external_id="local-1",
                audio_sha256=None,
                pcm_sha256=None,
            ),
        ),
        smart_turn_records=(),
        external_repository="pipecat-ai/smart-turn-data-v3.2-train",
        external_revision="2" * 40,
    )

    assert report.unverified_local_record_count == 1
    assert not report.clean_comparative_claim_permitted
