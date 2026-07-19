from __future__ import annotations

from collections import Counter

from app.training.tool_use.scenario import (
    AssistantResponseMode,
    PlannedOutcome,
    ScenarioGenerationMode,
    ScenarioSamplingProfile,
    ScenarioSpec,
    SearchFlow,
    SegmentBucket,
    UtteranceForm,
    sample_scenarios,
)


def test_scenario_sampling_is_reproducible_and_group_split_is_stable() -> None:
    first = sample_scenarios(
        count=500,
        random_seed=17,
        profile=ScenarioSamplingProfile.STANDARD,
    )
    second = sample_scenarios(
        count=500,
        random_seed=17,
        profile=ScenarioSamplingProfile.STANDARD,
    )

    assert first == second
    split_by_group = {scenario.leakage_group_id: scenario.split for scenario in first}
    assert all(split_by_group[scenario.leakage_group_id] is scenario.split for scenario in first)
    assert all(
        ScenarioSpec.model_validate_json(scenario.model_dump_json()) == scenario
        for scenario in first
    )


def test_teacher_led_profile_supplies_only_a_loose_four_turn_brief() -> None:
    scenarios = sample_scenarios(
        count=24,
        random_seed=19,
        profile=ScenarioSamplingProfile.TEACHER_LED_TOOL_USE,
    )

    assert all(
        scenario.generation_mode is ScenarioGenerationMode.TEACHER_LED for scenario in scenarios
    )
    assert all(scenario.family == "teacher_led_tool_use" for scenario in scenarios)
    assert all(len(scenario.turns) == 4 for scenario in scenarios)
    assert all(not turn.tool_steps for scenario in scenarios for turn in scenario.turns)
    assert len({scenario.topic for scenario in scenarios}) >= 8


def test_scenario_sampling_approximates_approved_tool_round_distribution() -> None:
    scenarios = sample_scenarios(
        count=10_000,
        random_seed=31,
        profile=ScenarioSamplingProfile.STANDARD,
    )
    round_counts = Counter(
        max(len(turn.tool_steps) for turn in scenario.turns) for scenario in scenarios
    )

    assert 0.32 <= round_counts[0] / len(scenarios) <= 0.38
    assert 0.44 <= round_counts[1] / len(scenarios) <= 0.50
    assert 0.15 <= round_counts[2] / len(scenarios) <= 0.21
    assert round_counts[3] == 0
    assert all(len(turn.tool_steps) <= 3 for scenario in scenarios for turn in scenario.turns)


def test_scenario_sampling_delays_tool_need_and_limits_direct_questions() -> None:
    scenarios = sample_scenarios(
        count=10_000,
        random_seed=37,
        profile=ScenarioSamplingProfile.STANDARD,
    )
    multi_turn_tool_scenarios = tuple(
        scenario
        for scenario in scenarios
        if len(scenario.turns) > 1 and any(turn.tool_steps for turn in scenario.turns)
    )
    delayed_tool_scenarios = tuple(
        scenario for scenario in multi_turn_tool_scenarios if not scenario.turns[0].tool_steps
    )
    utterance_forms = tuple(
        turn.utterance_form for scenario in scenarios for turn in scenario.turns
    )

    delayed_ratio = len(delayed_tool_scenarios) / len(multi_turn_tool_scenarios)
    direct_question_ratio = utterance_forms.count(UtteranceForm.DIRECT_QUESTION) / len(
        utterance_forms
    )
    pre_tool_turns = tuple(
        turn
        for scenario in scenarios
        for turn in scenario.turns[
            : next(
                (
                    index
                    for index, candidate_turn in enumerate(scenario.turns)
                    if candidate_turn.tool_steps
                ),
                0,
            )
        ]
    )

    assert 0.20 <= delayed_ratio <= 0.40
    assert direct_question_ratio <= 0.15
    assert all(
        turn.utterance_form not in (UtteranceForm.DIRECT_QUESTION, UtteranceForm.REQUEST)
        for turn in pre_tool_turns
    )
    assert all(
        scenario.turns[0].tool_steps
        for scenario in multi_turn_tool_scenarios
        if any(
            step.tool_name.value == "get_time"
            for turn in scenario.turns
            for step in turn.tool_steps
        )
    )


def test_no_tool_follow_ups_forbid_new_information_needs() -> None:
    scenarios = sample_scenarios(
        count=1_000,
        random_seed=43,
        profile=ScenarioSamplingProfile.STANDARD,
    )
    no_tool_follow_ups = tuple(
        turn
        for scenario in scenarios
        for first_tool_index in (
            next(
                (
                    index
                    for index, candidate_turn in enumerate(scenario.turns)
                    if candidate_turn.tool_steps
                ),
                0,
            ),
        )
        for turn_index, turn in enumerate(scenario.turns)
        if turn_index > first_tool_index and not turn.tool_steps
    )

    assert no_tool_follow_ups
    assert all(
        "Do not introduce a new fact, calculation, current-information need, or external action."
        in turn.user_instruction
        for turn in no_tool_follow_ups
    )


def test_no_tool_scenarios_are_short_and_have_explicit_task_openings() -> None:
    scenarios = sample_scenarios(
        count=10_000,
        random_seed=47,
        profile=ScenarioSamplingProfile.STANDARD,
    )
    no_tool_scenarios = tuple(
        scenario for scenario in scenarios if not any(turn.tool_steps for turn in scenario.turns)
    )

    assert no_tool_scenarios
    assert all(
        scenario.family not in ("ordinary_dialogue", "provided_context")
        for scenario in no_tool_scenarios
    )
    assert all(len(scenario.turns) <= 3 for scenario in no_tool_scenarios)
    assert all(
        scenario.turns[0].utterance_form not in (UtteranceForm.FRAGMENT, UtteranceForm.REACTION)
        for scenario in no_tool_scenarios
    )
    assert all(
        len(scenario.turns) == 1
        for scenario in no_tool_scenarios
        if scenario.family == "stable_knowledge"
    )


def test_long_mixed_profile_places_tools_later_in_eight_to_twelve_rounds() -> None:
    scenarios = sample_scenarios(
        count=1_000,
        random_seed=53,
        profile=ScenarioSamplingProfile.LONG_MIXED,
    )
    first_tool_indexes = tuple(
        next(index for index, turn in enumerate(scenario.turns) if turn.tool_steps)
        for scenario in scenarios
    )

    assert all(scenario.family == "long_mixed" for scenario in scenarios)
    assert all(scenario.length_band.value == "long" for scenario in scenarios)
    assert all(8 <= len(scenario.turns) <= 12 for scenario in scenarios)
    assert all(index >= 2 for index in first_tool_indexes)
    assert len(set(first_tool_indexes)) >= 3
    assert all(
        1 <= sum(bool(turn.tool_steps) for turn in scenario.turns) <= 2 for scenario in scenarios
    )
    assert all(len(turn.tool_steps) <= 2 for scenario in scenarios for turn in scenario.turns)


def test_bucket_calibration_balances_behavior_and_speech_style() -> None:
    scenarios = sample_scenarios(
        count=40,
        random_seed=59,
        profile=ScenarioSamplingProfile.BUCKET_CALIBRATION,
    )
    family_counts = Counter(scenario.family for scenario in scenarios)

    assert family_counts == Counter({bucket.value: 5 for bucket in SegmentBucket})
    assert tuple(scenario.family for scenario in scenarios[:8]) == tuple(
        bucket.value for bucket in SegmentBucket
    )
    assert all(2 <= len(scenario.turns) <= 4 for scenario in scenarios)
    for bucket in SegmentBucket:
        bucket_scenarios = tuple(
            scenario for scenario in scenarios if scenario.family == bucket.value
        )
        assert len({scenario.speech_style for scenario in bucket_scenarios}) == 5
    sequential = tuple(
        scenario
        for scenario in scenarios
        if scenario.family == SegmentBucket.SEQUENTIAL_TOOLS.value
    )
    recovery = tuple(
        scenario
        for scenario in scenarios
        if scenario.family == SegmentBucket.CORRECTION_RECOVERY.value
    )
    search = tuple(
        scenario for scenario in scenarios if scenario.family == SegmentBucket.SEARCH.value
    )
    assert all(len(scenario.turns[0].tool_steps) == 2 for scenario in sequential)
    assert all(sum(len(turn.tool_steps) for turn in scenario.turns) == 2 for scenario in recovery)
    assert (
        sum(
            scenario.turns[0].response_mode is AssistantResponseMode.CLARIFY_SEARCH
            for scenario in search
        )
        == 2
    )


def test_bucket_calibration_requires_an_equal_bucket_count() -> None:
    try:
        sample_scenarios(
            count=41,
            random_seed=61,
            profile=ScenarioSamplingProfile.BUCKET_CALIBRATION,
        )
    except ValueError as error:
        assert str(error) == "Bucket calibration count must be divisible by 8."
    else:
        raise AssertionError("Unequal bucket calibration count was accepted.")


def test_search_calibration_balances_direct_clarified_and_refined_flows() -> None:
    scenarios = sample_scenarios(
        count=15,
        random_seed=71,
        profile=ScenarioSamplingProfile.SEARCH_CALIBRATION,
    )

    assert Counter(scenario.family for scenario in scenarios) == Counter(
        {flow.value: 5 for flow in SearchFlow}
    )
    for flow in SearchFlow:
        flow_scenarios = tuple(scenario for scenario in scenarios if scenario.family == flow.value)
        assert len({scenario.speech_style for scenario in flow_scenarios}) == 5
    clarified = tuple(
        scenario
        for scenario in scenarios
        if scenario.family == SearchFlow.CLARIFY_THEN_SEARCH.value
    )
    assert all(
        tuple(turn.response_mode for turn in scenario.turns)
        == (
            AssistantResponseMode.CLARIFY_SEARCH,
            AssistantResponseMode.ANSWER,
        )
        for scenario in clarified
    )
    assert all(
        tuple(step.tool_name.value for turn in scenario.turns for step in turn.tool_steps)
        == ("search",)
        for scenario in clarified
    )


def test_scenario_sampling_makes_adverse_multi_call_outcomes_rare() -> None:
    scenarios = sample_scenarios(
        count=100_000,
        random_seed=41,
        profile=ScenarioSamplingProfile.STANDARD,
    )
    one_call_turns = tuple(
        turn for scenario in scenarios for turn in scenario.turns if len(turn.tool_steps) == 1
    )
    multi_call_turns = tuple(
        turn for scenario in scenarios for turn in scenario.turns if len(turn.tool_steps) > 1
    )

    one_call_adverse_ratio = sum(
        turn.tool_steps[-1].outcome is not PlannedOutcome.SUCCESS for turn in one_call_turns
    ) / len(one_call_turns)
    multi_call_adverse_ratio = sum(
        turn.tool_steps[-1].outcome is not PlannedOutcome.SUCCESS for turn in multi_call_turns
    ) / len(multi_call_turns)

    assert one_call_adverse_ratio <= 0.04
    assert multi_call_adverse_ratio <= 0.01
