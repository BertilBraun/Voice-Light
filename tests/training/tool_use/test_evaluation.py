from __future__ import annotations

from app.training.tool_use.evaluation import (
    EvaluationCaseKind,
    HeldoutGenerationCase,
    ModelVariant,
    generation_metrics,
    parse_generation_prediction,
)


def test_generation_prediction_parses_spoken_bridge_and_valid_tool_call() -> None:
    case = HeldoutGenerationCase(
        case_id="record-1-assistant-2",
        record_id="record-1",
        kind=EvaluationCaseKind.TOOL_CALL,
        tools=(),
        prefix_messages=(),
        expected_tool_name="search",
    )

    prediction = parse_generation_prediction(
        case=case,
        model_variant=ModelVariant.ADAPTER,
        random_seed=17,
        generated_text=(
            "Let me check that."
            '<tool_call>{"name":"search","arguments":{"query":"current train time"}}</tool_call>'
        ),
    )

    assert prediction.parser_valid
    assert prediction.tool_schema_valid
    assert prediction.expected_tool_name == "search"
    assert prediction.generated_tool_name == "search"
    assert prediction.spoken_text == "Let me check that."
    assert prediction.bridge_word_count == 4


def test_generation_metrics_distinguish_tool_no_tool_and_continuation_behavior() -> None:
    tool_case = HeldoutGenerationCase(
        case_id="tool",
        record_id="record-1",
        kind=EvaluationCaseKind.TOOL_CALL,
        tools=(),
        prefix_messages=(),
        expected_tool_name="calculate",
    )
    no_tool_case = HeldoutGenerationCase(
        case_id="no-tool",
        record_id="record-2",
        kind=EvaluationCaseKind.NO_TOOL,
        tools=(),
        prefix_messages=(),
        expected_tool_name=None,
    )
    continuation_case = HeldoutGenerationCase(
        case_id="continuation",
        record_id="record-3",
        kind=EvaluationCaseKind.TOOL_CONTINUATION,
        tools=(),
        prefix_messages=(),
        expected_tool_name=None,
    )
    predictions = (
        parse_generation_prediction(
            case=tool_case,
            model_variant=ModelVariant.BASE,
            random_seed=17,
            generated_text=(
                "I'll calculate it."
                '<tool_call>{"name":"calculate","arguments":{"expression":"6 * 7"}}</tool_call>'
            ),
        ),
        parse_generation_prediction(
            case=no_tool_case,
            model_variant=ModelVariant.BASE,
            random_seed=17,
            generated_text="That sounds like the useful part.",
        ),
        parse_generation_prediction(
            case=continuation_case,
            model_variant=ModelVariant.BASE,
            random_seed=17,
            generated_text="It comes to 42.",
        ),
    )

    metrics = generation_metrics(predictions)

    assert metrics.tool_decision_accuracy == 1.0
    assert metrics.tool_name_accuracy == 1.0
    assert metrics.tool_schema_valid_rate == 1.0
    assert metrics.bridge_present_rate == 1.0
    assert metrics.bridge_concise_rate == 1.0
    assert metrics.no_tool_accuracy == 1.0
    assert metrics.continuation_nonempty_rate == 1.0
