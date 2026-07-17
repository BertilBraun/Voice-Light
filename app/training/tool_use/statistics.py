from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import Field

from app.training.tool_use.schema import (
    AssistantMessage,
    DatasetSplit,
    SuccessOutcome,
    ToolDialogueRecord,
    ToolResultMessage,
    ToolUseBaseModel,
    UserMessage,
)


class CountMetric(ToolUseBaseModel):
    label: str
    count: int = Field(ge=0)


class DatasetStatistics(ToolUseBaseModel):
    record_count: int = Field(ge=0)
    user_message_count: int = Field(ge=0)
    assistant_message_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    successful_tool_result_count: int = Field(ge=0)
    split_counts: tuple[CountMetric, ...]
    family_counts: tuple[CountMetric, ...]
    tool_counts: tuple[CountMetric, ...]
    exact_bridge_counts: tuple[CountMetric, ...]


def dataset_statistics(records: Sequence[ToolDialogueRecord]) -> DatasetStatistics:
    split_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    bridge_counter: Counter[str] = Counter()
    user_message_count = 0
    assistant_message_count = 0
    successful_tool_result_count = 0

    for record in records:
        split_counter[record.metadata.split.name.value] += 1
        family_counter[record.metadata.scenario.family] += 1
        for message in record.messages:
            match message:
                case AssistantMessage(audible_text=audible_text, tool_calls=tool_calls):
                    assistant_message_count += 1
                    for tool_call in tool_calls:
                        tool_counter[tool_call.tool_name] += 1
                    if tool_calls:
                        bridge_counter[_normalize_phrase(audible_text)] += 1
                case ToolResultMessage(outcome=SuccessOutcome()):
                    successful_tool_result_count += 1
                case UserMessage():
                    user_message_count += 1
                case _:
                    pass

    return DatasetStatistics(
        record_count=len(records),
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        tool_call_count=sum(tool_counter.values()),
        successful_tool_result_count=successful_tool_result_count,
        split_counts=_count_metrics(split_counter, tuple(split.value for split in DatasetSplit)),
        family_counts=_count_metrics(family_counter),
        tool_counts=_count_metrics(tool_counter),
        exact_bridge_counts=_count_metrics(bridge_counter),
    )


def _normalize_phrase(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _count_metrics(
    counter: Counter[str],
    required_labels: tuple[str, ...] = (),
) -> tuple[CountMetric, ...]:
    labels = set(counter)
    labels.update(required_labels)
    return tuple(CountMetric(label=label, count=counter[label]) for label in sorted(labels))
