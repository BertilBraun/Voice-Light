from __future__ import annotations

from collections import Counter

from app.training.tool_use.scenario import (
    PlannedOutcome,
    ScenarioSpec,
    UtteranceForm,
    sample_scenarios,
)


def test_scenario_sampling_is_reproducible_and_group_split_is_stable() -> None:
    first = sample_scenarios(count=500, random_seed=17)
    second = sample_scenarios(count=500, random_seed=17)

    assert first == second
    split_by_group = {scenario.leakage_group_id: scenario.split for scenario in first}
    assert all(split_by_group[scenario.leakage_group_id] is scenario.split for scenario in first)
    assert all(
        ScenarioSpec.model_validate_json(scenario.model_dump_json()) == scenario
        for scenario in first
    )


def test_scenario_sampling_approximates_approved_tool_round_distribution() -> None:
    scenarios = sample_scenarios(count=10_000, random_seed=31)
    round_counts = Counter(
        max(len(turn.tool_steps) for turn in scenario.turns) for scenario in scenarios
    )

    assert 0.32 <= round_counts[0] / len(scenarios) <= 0.38
    assert 0.42 <= round_counts[1] / len(scenarios) <= 0.48
    assert 0.15 <= round_counts[2] / len(scenarios) <= 0.21
    assert 0.01 <= round_counts[3] / len(scenarios) <= 0.03
    assert all(len(turn.tool_steps) <= 3 for scenario in scenarios for turn in scenario.turns)


def test_scenario_sampling_delays_tool_need_and_limits_direct_questions() -> None:
    scenarios = sample_scenarios(count=10_000, random_seed=37)
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

    assert 0.38 <= delayed_ratio <= 0.48
    assert direct_question_ratio <= 0.15


def test_scenario_sampling_makes_adverse_multi_call_outcomes_rare() -> None:
    scenarios = sample_scenarios(count=100_000, random_seed=41)
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
