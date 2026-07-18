from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import model_validator

from app.training.tool_use.schema import (
    DatasetSplit,
    LengthBand,
    SpeechStyle,
    ToolUseBaseModel,
)

ChoiceType = TypeVar("ChoiceType")


class ToolName(StrEnum):
    SEARCH = "search"
    CALCULATE = "calculate"
    GET_TIME = "get_time"


class PlannedOutcome(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILURE = "failure"
    TIMEOUT = "timeout"


class FollowUpKind(StrEnum):
    NONE = "none"
    PRONOUN = "pronoun"
    CORRECTION = "correction"
    CONSTRAINT = "constraint"
    CLARIFICATION = "clarification"
    DELAYED_REQUEST = "delayed_request"


class UtteranceForm(StrEnum):
    DIRECT_QUESTION = "direct_question"
    REQUEST = "request"
    CONTEXT_FIRST = "context_first"
    FRAGMENT = "fragment"
    REACTION = "reaction"
    SELF_REPAIR = "self_repair"


class PlannedToolStep(ToolUseBaseModel):
    tool_name: ToolName
    outcome: PlannedOutcome = PlannedOutcome.SUCCESS


class AssistantTurnPlan(ToolUseBaseModel):
    user_instruction: str
    tool_steps: tuple[PlannedToolStep, ...]
    follow_up_kind: FollowUpKind
    utterance_form: UtteranceForm


class ScenarioSpec(ToolUseBaseModel):
    schema_version: Literal["voice-light.scenario/v1"] = "voice-light.scenario/v1"
    scenario_id: str
    random_seed: int
    family: str
    topic: str
    length_band: LengthBand
    speech_style: SpeechStyle
    turns: tuple[AssistantTurnPlan, ...]
    split: DatasetSplit
    leakage_group_id: str

    @model_validator(mode="after")
    def validate_turns(self) -> ScenarioSpec:
        if not self.turns:
            raise ValueError("A scenario must contain at least one user turn.")
        if any(len(turn.tool_steps) > 3 for turn in self.turns):
            raise ValueError("A scenario turn may contain at most three sequential tool calls.")
        return self


@dataclass(frozen=True)
class ScenarioTemplate:
    family: str
    topic: str
    user_instruction: str
    tool_names: tuple[ToolName, ...]


NO_TOOL_TEMPLATES = (
    ScenarioTemplate(
        family="ordinary_dialogue",
        topic="daily life",
        user_instruction=(
            "Start a brief exchange about a preference, minor decision, or everyday situation "
            "that needs no tool."
        ),
        tool_names=(),
    ),
    ScenarioTemplate(
        family="provided_context",
        topic="user-provided information",
        user_instruction=(
            "Mention a small fact from the user's situation and continue in a way the assistant "
            "can answer entirely from that fact."
        ),
        tool_names=(),
    ),
    ScenarioTemplate(
        family="stable_knowledge",
        topic="common knowledge",
        user_instruction=(
            "Bring up a simple stable-knowledge topic in a conversational way that needs no tool."
        ),
        tool_names=(),
    ),
)

ONE_TOOL_TEMPLATES = (
    ScenarioTemplate(
        family="current_lookup",
        topic="recent public information",
        user_instruction="Naturally request one current or externally verifiable fact.",
        tool_names=(ToolName.SEARCH,),
    ),
    ScenarioTemplate(
        family="specific_lookup",
        topic="event or organization detail",
        user_instruction="Naturally request a specific detail that should be looked up.",
        tool_names=(ToolName.SEARCH,),
    ),
    ScenarioTemplate(
        family="precise_arithmetic",
        topic="basic practical arithmetic",
        user_instruction=(
            "Give a short practical arithmetic expression and request its exact result."
        ),
        tool_names=(ToolName.CALCULATE,),
    ),
    ScenarioTemplate(
        family="current_local_time",
        topic="current date or local time",
        user_instruction="Work the current local date or time into a natural request.",
        tool_names=(ToolName.GET_TIME,),
    ),
)

TWO_TOOL_TEMPLATES = (
    ScenarioTemplate(
        family="lookup_then_calculate",
        topic="a looked-up number followed by arithmetic",
        user_instruction=(
            "Naturally request a result that needs one looked-up number and one simple calculation."
        ),
        tool_names=(ToolName.SEARCH, ToolName.CALCULATE),
    ),
    ScenarioTemplate(
        family="time_then_calculate",
        topic="elapsed or remaining time",
        user_instruction=(
            "Naturally request a simple elapsed or remaining time that needs the current time and "
            "one calculation."
        ),
        tool_names=(ToolName.GET_TIME, ToolName.CALCULATE),
    ),
    ScenarioTemplate(
        family="refined_lookup",
        topic="two-step information lookup",
        user_instruction=(
            "Naturally request one fact and a closely related detail that depends on it."
        ),
        tool_names=(ToolName.SEARCH, ToolName.SEARCH),
    ),
)

THREE_TOOL_TEMPLATES = (
    ScenarioTemplate(
        family="lookup_time_calculate",
        topic="simple date comparison",
        user_instruction=(
            "Naturally request a concise date difference requiring one lookup, the current time, "
            "and one basic calculation."
        ),
        tool_names=(ToolName.SEARCH, ToolName.GET_TIME, ToolName.CALCULATE),
    ),
)

TOPIC_VARIANTS = (
    "books",
    "films",
    "music",
    "sports",
    "travel",
    "public transport",
    "shops",
    "museums",
    "food",
    "technology",
    "nature",
    "local events",
    "work",
    "household tasks",
    "personal scheduling",
)

SPEECH_STYLE_WEIGHTS: tuple[tuple[SpeechStyle, float], ...] = (
    (SpeechStyle.CLEAN, 0.25),
    (SpeechStyle.CASUAL, 0.35),
    (SpeechStyle.ASR_FRAGMENT, 0.20),
    (SpeechStyle.REPAIR, 0.15),
    (SpeechStyle.FORMAL, 0.05),
)

LENGTH_BAND_WEIGHTS: tuple[tuple[LengthBand, float], ...] = (
    (LengthBand.SHORT, 0.25),
    (LengthBand.MEDIUM, 0.55),
    (LengthBand.LONG, 0.20),
)

OPENING_FORM_WEIGHTS: tuple[tuple[UtteranceForm, float], ...] = (
    (UtteranceForm.DIRECT_QUESTION, 0.12),
    (UtteranceForm.REQUEST, 0.28),
    (UtteranceForm.CONTEXT_FIRST, 0.35),
    (UtteranceForm.FRAGMENT, 0.15),
    (UtteranceForm.SELF_REPAIR, 0.10),
)

FOLLOW_UP_FORM_WEIGHTS: tuple[tuple[UtteranceForm, float], ...] = (
    (UtteranceForm.DIRECT_QUESTION, 0.06),
    (UtteranceForm.REQUEST, 0.16),
    (UtteranceForm.CONTEXT_FIRST, 0.16),
    (UtteranceForm.FRAGMENT, 0.25),
    (UtteranceForm.REACTION, 0.27),
    (UtteranceForm.SELF_REPAIR, 0.10),
)


def sample_scenarios(count: int, random_seed: int) -> tuple[ScenarioSpec, ...]:
    if count <= 0:
        raise ValueError("Scenario count must be positive.")
    generator = random.Random(random_seed)
    scenarios: list[ScenarioSpec] = []
    for index in range(count):
        scenario_seed = generator.randrange(0, 2**31)
        scenario_generator = random.Random(scenario_seed)
        template = _sample_template(scenario_generator)
        length_band = _weighted_choice(scenario_generator, LENGTH_BAND_WEIGHTS)
        speech_style = _weighted_choice(scenario_generator, SPEECH_STYLE_WEIGHTS)
        topic_variant = scenario_generator.choice(TOPIC_VARIANTS)
        topic = f"{template.topic}: {topic_variant}"
        turns = _build_turns(
            template=template,
            length_band=length_band,
            generator=scenario_generator,
        )
        leakage_group_id = _leakage_group(template, topic_variant)
        scenarios.append(
            ScenarioSpec(
                scenario_id=f"scenario-{random_seed}-{index:06d}",
                random_seed=scenario_seed,
                family=template.family,
                topic=topic,
                length_band=length_band,
                speech_style=speech_style,
                turns=turns,
                split=_split_for_group(leakage_group_id),
                leakage_group_id=leakage_group_id,
            )
        )
    return tuple(scenarios)


def write_scenarios(path: Path, scenarios: tuple[ScenarioSpec, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(scenario.model_dump_json() for scenario in scenarios)
    path.write_text(f"{content}\n", encoding="utf-8")


def read_scenarios(path: Path) -> tuple[ScenarioSpec, ...]:
    scenarios: list[ScenarioSpec] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scenarios.append(ScenarioSpec.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"Invalid scenario on line {line_number}: {error}") from error
    return tuple(scenarios)


def _sample_template(generator: random.Random) -> ScenarioTemplate:
    round_band = generator.choices(
        population=(0, 1, 2, 3),
        weights=(0.35, 0.45, 0.18, 0.02),
        k=1,
    )[0]
    match round_band:
        case 0:
            return generator.choice(NO_TOOL_TEMPLATES)
        case 1:
            return generator.choice(ONE_TOOL_TEMPLATES)
        case 2:
            return generator.choice(TWO_TOOL_TEMPLATES)
        case 3:
            return generator.choice(THREE_TOOL_TEMPLATES)
        case _:
            raise AssertionError("random.choices returned an excluded tool-round band.")


def _build_turns(
    template: ScenarioTemplate,
    length_band: LengthBand,
    generator: random.Random,
) -> tuple[AssistantTurnPlan, ...]:
    match length_band:
        case LengthBand.SHORT:
            turn_count = 1
        case LengthBand.MEDIUM:
            turn_count = generator.choice((2, 3))
        case LengthBand.LONG:
            turn_count = generator.choice((4, 5))
    tool_turn_index = _tool_turn_index(template, turn_count, generator)
    turns: list[AssistantTurnPlan] = []
    follow_up_kinds = (
        FollowUpKind.PRONOUN,
        FollowUpKind.CORRECTION,
        FollowUpKind.CONSTRAINT,
        FollowUpKind.CLARIFICATION,
    )
    for turn_index in range(turn_count):
        if turn_index < tool_turn_index:
            turns.append(
                AssistantTurnPlan(
                    user_instruction=_pre_tool_instruction(turn_index),
                    tool_steps=(),
                    follow_up_kind=(
                        FollowUpKind.NONE if turn_index == 0 else generator.choice(follow_up_kinds)
                    ),
                    utterance_form=_sample_utterance_form(
                        generator,
                        is_follow_up=turn_index > 0,
                    ),
                )
            )
            continue
        if turn_index == tool_turn_index:
            turns.append(
                AssistantTurnPlan(
                    user_instruction=template.user_instruction,
                    tool_steps=_planned_tool_steps(template.tool_names, generator),
                    follow_up_kind=(
                        FollowUpKind.NONE if turn_index == 0 else FollowUpKind.DELAYED_REQUEST
                    ),
                    utterance_form=_sample_utterance_form(
                        generator,
                        is_follow_up=turn_index > 0,
                    ),
                )
            )
            continue
        follow_up_kind = generator.choice(follow_up_kinds)
        correction_requires_search = (
            follow_up_kind is FollowUpKind.CORRECTION
            and ToolName.SEARCH in template.tool_names
            and generator.random() < 0.5
        )
        turns.append(
            AssistantTurnPlan(
                user_instruction=_follow_up_instruction(follow_up_kind),
                tool_steps=(
                    (PlannedToolStep(tool_name=ToolName.SEARCH),)
                    if correction_requires_search
                    else ()
                ),
                follow_up_kind=follow_up_kind,
                utterance_form=_sample_follow_up_form(generator, follow_up_kind),
            )
        )
    return tuple(turns)


def _tool_turn_index(
    template: ScenarioTemplate,
    turn_count: int,
    generator: random.Random,
) -> int:
    if not template.tool_names or turn_count == 1 or generator.random() >= 0.43:
        return 0
    return generator.randrange(1, min(turn_count, 3))


def _pre_tool_instruction(turn_index: int) -> str:
    if turn_index == 0:
        return (
            "Begin with a related personal detail, preference, or observation that the assistant "
            "can respond to without a tool. Do not request the planned current fact yet."
        )
    return (
        "React naturally and add one useful detail that sets up the later request without needing "
        "a tool yet."
    )


def _sample_utterance_form(
    generator: random.Random,
    is_follow_up: bool,
) -> UtteranceForm:
    return _weighted_choice(
        generator,
        FOLLOW_UP_FORM_WEIGHTS if is_follow_up else OPENING_FORM_WEIGHTS,
    )


def _sample_follow_up_form(
    generator: random.Random,
    follow_up_kind: FollowUpKind,
) -> UtteranceForm:
    match follow_up_kind:
        case FollowUpKind.PRONOUN:
            return generator.choice((UtteranceForm.FRAGMENT, UtteranceForm.REACTION))
        case FollowUpKind.CORRECTION:
            return generator.choice(
                (
                    UtteranceForm.SELF_REPAIR,
                    UtteranceForm.SELF_REPAIR,
                    UtteranceForm.CONTEXT_FIRST,
                )
            )
        case FollowUpKind.CONSTRAINT:
            return generator.choice((UtteranceForm.REQUEST, UtteranceForm.CONTEXT_FIRST))
        case FollowUpKind.CLARIFICATION:
            return generator.choice(
                (
                    UtteranceForm.REACTION,
                    UtteranceForm.FRAGMENT,
                    UtteranceForm.DIRECT_QUESTION,
                )
            )
        case FollowUpKind.NONE | FollowUpKind.DELAYED_REQUEST:
            return _sample_utterance_form(generator, is_follow_up=True)


def _follow_up_instruction(follow_up_kind: FollowUpKind) -> str:
    match follow_up_kind:
        case FollowUpKind.PRONOUN:
            return (
                "Refer back with a pronoun or elliptical phrase as part of a natural reaction. "
                "Avoid a standalone 'What about that?'"
            )
        case FollowUpKind.CORRECTION:
            return (
                "Correct or revise one detail naturally, possibly mid-sentence, and continue from "
                "the correction."
            )
        case FollowUpKind.CONSTRAINT:
            return (
                "Add one simple preference or constraint as a continuation rather than starting "
                "a fresh formal question."
            )
        case FollowUpKind.CLARIFICATION:
            return (
                "React to one specific part of the preceding answer and clarify what the user "
                "meant. Avoid a generic follow-up."
            )
        case FollowUpKind.NONE | FollowUpKind.DELAYED_REQUEST:
            raise ValueError("A generated follow-up cannot use FollowUpKind.NONE.")


def _weighted_choice(
    generator: random.Random,
    weighted_values: tuple[tuple[ChoiceType, float], ...],
) -> ChoiceType:
    values = tuple(value for value, _ in weighted_values)
    weights = tuple(weight for _, weight in weighted_values)
    return generator.choices(population=values, weights=weights, k=1)[0]


def _planned_tool_steps(
    tool_names: tuple[ToolName, ...],
    generator: random.Random,
) -> tuple[PlannedToolStep, ...]:
    if not tool_names:
        return ()
    adverse_probability = _adverse_probability(tool_names)
    final_outcome = (
        generator.choices(
            population=(
                PlannedOutcome.EMPTY,
                PlannedOutcome.FAILURE,
                PlannedOutcome.TIMEOUT,
            ),
            weights=(0.35, 0.40, 0.25),
            k=1,
        )[0]
        if generator.random() < adverse_probability
        else PlannedOutcome.SUCCESS
    )
    return tuple(
        PlannedToolStep(
            tool_name=tool_name,
            outcome=final_outcome if index == len(tool_names) - 1 else PlannedOutcome.SUCCESS,
        )
        for index, tool_name in enumerate(tool_names)
    )


def _adverse_probability(tool_names: tuple[ToolName, ...]) -> float:
    if len(tool_names) > 1:
        return 0.005
    match tool_names[0]:
        case ToolName.SEARCH:
            return 0.04
        case ToolName.GET_TIME:
            return 0.01
        case ToolName.CALCULATE:
            return 0.005


def _leakage_group(template: ScenarioTemplate, topic_variant: str) -> str:
    normalized_topic = topic_variant.replace(" ", "-")
    tool_path = "-".join(tool_name.value for tool_name in template.tool_names) or "none"
    return f"{template.family}-{normalized_topic}-{tool_path}"


def _split_for_group(leakage_group_id: str) -> DatasetSplit:
    digest = hashlib.sha256(leakage_group_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:2], byteorder="big") % 100
    if bucket < 80:
        return DatasetSplit.TRAIN
    if bucket < 90:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST
