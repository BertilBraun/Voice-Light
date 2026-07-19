from __future__ import annotations

from app.training.tool_use.schema import DatasetSplit, SpeechStyle
from app.training.tool_use.training_data import prepare_training_corpus
from tests.training.tool_use.test_composition import _source_record
from tests.training.tool_use.test_renderer import PrefixStableTokenizer


def test_training_corpus_preserves_existing_source_splits() -> None:
    source_records = tuple(
        _source_record(
            record_id=f"{split.value}-{family}-{candidate_index}",
            family=family,
            speech_style=SpeechStyle.CLEAN,
            split=split,
        )
        for split in DatasetSplit
        for family in ("teacher_led_tool_use", "teacher_led_no_tool")
        for candidate_index in range(4)
    )

    corpus = prepare_training_corpus(
        tokenizer=PrefixStableTokenizer(),
        source_records=source_records,
        logical_epochs=2,
        random_seed=101,
        minimum_user_turns=8,
        maximum_user_turns=16,
        maximum_sequence_tokens=128,
    )

    assert corpus.source_records == source_records
    assert {plan.logical_epoch for plan in corpus.training.plans} == {0, 1}
    assert {plan.logical_epoch for plan in corpus.validation.plans} == {0}
    selected_ids = {
        record_id
        for plan in (*corpus.training.plans, *corpus.validation.plans)
        for record_id in plan.segment_record_ids
    }
    assert not any(record_id.startswith("test-") for record_id in selected_ids)
