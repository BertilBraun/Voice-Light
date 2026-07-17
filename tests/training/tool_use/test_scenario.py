from __future__ import annotations

from collections import Counter

from app.training.tool_use.scenario import ScenarioSpec, sample_scenarios


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
    round_counts = Counter(len(scenario.turns[0].tool_steps) for scenario in scenarios)

    assert 0.32 <= round_counts[0] / len(scenarios) <= 0.38
    assert 0.42 <= round_counts[1] / len(scenarios) <= 0.48
    assert 0.15 <= round_counts[2] / len(scenarios) <= 0.21
    assert 0.01 <= round_counts[3] / len(scenarios) <= 0.03
    assert all(len(turn.tool_steps) <= 3 for scenario in scenarios for turn in scenario.turns)
