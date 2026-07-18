from __future__ import annotations

from collections import Counter

from app.training.tool_use.scenario import SegmentBucket
from app.training.tool_use.schema import DatasetSplit, SpeechStyle
from app.training.tool_use.training_data import assign_proof_of_concept_splits
from tests.training.tool_use.test_composition import _source_record


def test_proof_of_concept_split_holds_out_two_records_per_bucket_and_style() -> None:
    source_records = tuple(
        _source_record(
            record_id=f"{bucket.value}-{speech_style.value}-{candidate_index}",
            family=bucket.value,
            speech_style=speech_style,
            split=DatasetSplit.TEST,
        )
        for bucket in SegmentBucket
        for speech_style in SpeechStyle
        for candidate_index in range(3)
    )

    assigned = assign_proof_of_concept_splits(
        source_records=source_records,
        holdout_record_count=80,
        random_seed=101,
    )
    repeated = assign_proof_of_concept_splits(
        source_records=source_records,
        holdout_record_count=80,
        random_seed=101,
    )

    assert assigned == repeated
    assert sum(record.metadata.split.name is DatasetSplit.TRAIN for record in assigned) == 40
    validation_records = tuple(
        record for record in assigned if record.metadata.split.name is DatasetSplit.VALIDATION
    )
    assert len(validation_records) == 80
    assert all(record.metadata.split.name is not DatasetSplit.TEST for record in assigned)
    assert Counter(
        (
            record.metadata.scenario.family,
            record.metadata.scenario.speech_style,
        )
        for record in validation_records
    ) == Counter(
        {
            (bucket.value, speech_style): 2
            for bucket in SegmentBucket
            for speech_style in SpeechStyle
        }
    )
