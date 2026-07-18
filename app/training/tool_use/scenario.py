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


class AssistantResponseMode(StrEnum):
    ANSWER = "answer"
    CLARIFY_SEARCH = "clarify_search"
    CONFIRM_SEARCH = "confirm_search"


class ScenarioSamplingProfile(StrEnum):
    STANDARD = "standard"
    LONG_MIXED = "long_mixed"
    BUCKET_CALIBRATION = "bucket_calibration"
    SEARCH_CALIBRATION = "search_calibration"


class SegmentBucket(StrEnum):
    NATURAL_CONVERSATION = "natural_conversation"
    DRAFTING = "drafting"
    BRAINSTORM_DECISION = "brainstorm_decision"
    SEARCH = "search"
    CALCULATE = "calculate"
    GET_TIME = "get_time"
    SEQUENTIAL_TOOLS = "sequential_tools"
    CORRECTION_RECOVERY = "correction_recovery"


class SearchFlow(StrEnum):
    DIRECT = "search_direct"
    CLARIFY_THEN_SEARCH = "search_clarify_then_call"
    REFINE_THEN_SEARCH = "search_refine_then_call"


class PlannedToolStep(ToolUseBaseModel):
    tool_name: ToolName
    outcome: PlannedOutcome = PlannedOutcome.SUCCESS


class AssistantTurnPlan(ToolUseBaseModel):
    user_instruction: str
    tool_steps: tuple[PlannedToolStep, ...]
    follow_up_kind: FollowUpKind
    utterance_form: UtteranceForm
    response_mode: AssistantResponseMode = AssistantResponseMode.ANSWER


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
        for turn_index, turn in enumerate(self.turns):
            if turn.response_mode is not AssistantResponseMode.ANSWER and turn.tool_steps:
                raise ValueError("A clarification turn cannot call a tool.")
            if turn.response_mode is not AssistantResponseMode.ANSWER and not any(
                step.tool_name is ToolName.SEARCH
                for future_turn in self.turns[turn_index + 1 :]
                for step in future_turn.tool_steps
            ):
                raise ValueError("A search clarification must precede a later search call.")
        return self


@dataclass(frozen=True)
class ScenarioTemplate:
    family: str
    topic: str
    user_instruction: str
    tool_names: tuple[ToolName, ...]


@dataclass(frozen=True)
class SearchScenarioAxis:
    topic: str
    direct_request: str
    ambiguous_request: str
    clarification: str
    correction: str


SEARCH_SCENARIO_AXES = (
    SearchScenarioAxis(
        topic="live music listings",
        direct_request=(
            "Ask what live music is scheduled in Berlin on the evening of July 25, 2026."
        ),
        ambiguous_request=(
            "Ask what live music is scheduled on the evening of July 25, 2026, but deliberately "
            "omit the city. Do not imply a known current location."
        ),
        clarification="State that the intended city is Berlin.",
        correction="Correct the city from Berlin to Hamburg and request a revised lookup.",
    ),
    SearchScenarioAxis(
        topic="concert setlist",
        direct_request=(
            "Ask for the setlist from the Arctic Monkeys concert in Amsterdam on July 12, 2026."
        ),
        ambiguous_request=(
            "Ask for the Arctic Monkeys concert setlist from July 12, 2026, but deliberately omit "
            "which city the concert was in."
        ),
        clarification="State that the concert was in Amsterdam.",
        correction="Correct the concert city from Amsterdam to Rotterdam and request a new lookup.",
    ),
    SearchScenarioAxis(
        topic="specific product price",
        direct_request=("Ask for the current German price of Sony WH-1000XM5 headphones."),
        ambiguous_request=(
            "Ask for the current price of Sony headphones but deliberately omit the model. Do not "
            "name or imply any model number."
        ),
        clarification="State that the model is the Sony WH-1000XM5.",
        correction="Correct the model from WH-1000XM5 to WH-1000XM4 and request a revised lookup.",
    ),
    SearchScenarioAxis(
        topic="public transport status",
        direct_request="Ask whether Berlin's U8 line has delays today.",
        ambiguous_request=(
            "Ask whether a subway line has delays today but deliberately omit both the city and "
            "line name."
        ),
        clarification="State that the intended service is Berlin's U8 line.",
        correction="Correct the line from Berlin U8 to Berlin U6 and request a revised lookup.",
    ),
    SearchScenarioAxis(
        topic="cinema showtimes",
        direct_request=("Ask for tonight's showtimes for one named film at Zoo Palast in Berlin."),
        ambiguous_request=(
            "Ask for tonight's showtimes for one named film but deliberately omit the cinema and "
            "city."
        ),
        clarification="State that the cinema is Zoo Palast in Berlin.",
        correction=(
            "Correct the cinema from Zoo Palast to Kino International in Berlin and request a "
            "revised lookup."
        ),
    ),
)


NO_TOOL_TEMPLATES = (
    ScenarioTemplate(
        family="draft_revision",
        topic="rewriting user-provided text",
        user_instruction=(
            "Provide a short message draft and ask for one clear wording change such as warmer, "
            "shorter, or less formal. Ask only for drafted text, never to send the message. The "
            "assistant must need no external facts."
        ),
        tool_names=(),
    ),
    ScenarioTemplate(
        family="constrained_brainstorm",
        topic="creative suggestions from user constraints",
        user_instruction=(
            "Ask for two or three short creative suggestions for a personal or fictional item. "
            "Supply the theme and every needed constraint. Do not ask for factual recommendations "
            "about real products, places, services, or events."
        ),
        tool_names=(),
    ),
    ScenarioTemplate(
        family="personal_decision",
        topic="low-stakes advice from user-provided tradeoffs",
        user_instruction=(
            "Describe two low-stakes options and the relevant personal tradeoff, then ask for a "
            "brief opinion based only on those details. Do not ask about the assistant's own "
            "tastes or request external facts or actions."
        ),
        tool_names=(),
    ),
    ScenarioTemplate(
        family="stable_knowledge",
        topic="common knowledge",
        user_instruction=(
            "Ask about one genuinely timeless common-knowledge fact, definition, or established "
            "authorship. Avoid travel rules, time zones, venues, schedules, current roles, local "
            "facts, availability, prices, and recommendations."
        ),
        tool_names=(),
    ),
)

NATURAL_CONVERSATION_TEMPLATE = ScenarioTemplate(
    family="natural_conversation",
    topic="low-stakes personal conversation",
    user_instruction=(
        "Share one low-stakes personal moment, reaction, or feeling and invite a thoughtful "
        "conversational response. Do not request factual advice, hidden actions, therapy, or "
        "medical guidance. Give the assistant enough substance for a natural reply."
    ),
    tool_names=(),
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
            "Give a short practical arithmetic expression and request its exact numeric result. "
            "Avoid clock-time arithmetic, date arithmetic, and unit conversion."
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
            "Ask how many minutes remain until one explicit same-day deadline between 1:00 PM and "
            "11:00 PM. State the numeric deadline with AM or PM, avoid overnight spans, and "
            "require the current time followed by one simple subtraction."
        ),
        tool_names=(ToolName.GET_TIME, ToolName.CALCULATE),
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


def sample_scenarios(
    count: int,
    random_seed: int,
    profile: ScenarioSamplingProfile,
) -> tuple[ScenarioSpec, ...]:
    if count <= 0:
        raise ValueError("Scenario count must be positive.")
    bucket_count = len(SegmentBucket)
    if profile is ScenarioSamplingProfile.BUCKET_CALIBRATION and count % bucket_count != 0:
        raise ValueError(f"Bucket calibration count must be divisible by {bucket_count}.")
    search_flow_count = len(SearchFlow)
    if profile is ScenarioSamplingProfile.SEARCH_CALIBRATION and count % search_flow_count != 0:
        raise ValueError(f"Search calibration count must be divisible by {search_flow_count}.")
    generator = random.Random(random_seed)
    scenarios: list[ScenarioSpec] = []
    for index in range(count):
        scenario_seed = generator.randrange(0, 2**31)
        scenario_generator = random.Random(scenario_seed)
        if profile is ScenarioSamplingProfile.LONG_MIXED:
            scenarios.append(
                _sample_long_mixed_scenario(
                    index=index,
                    random_seed=random_seed,
                    scenario_seed=scenario_seed,
                    generator=scenario_generator,
                )
            )
            continue
        if profile is ScenarioSamplingProfile.BUCKET_CALIBRATION:
            scenarios.append(
                _sample_bucket_segment_scenario(
                    index=index,
                    count=count,
                    random_seed=random_seed,
                    scenario_seed=scenario_seed,
                    generator=scenario_generator,
                )
            )
            continue
        if profile is ScenarioSamplingProfile.SEARCH_CALIBRATION:
            scenarios.append(
                _sample_search_calibration_scenario(
                    index=index,
                    count=count,
                    random_seed=random_seed,
                    scenario_seed=scenario_seed,
                )
            )
            continue
        template = _sample_template(scenario_generator)
        length_band = _sample_length_band(template, scenario_generator)
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


def _sample_search_calibration_scenario(
    index: int,
    count: int,
    random_seed: int,
    scenario_seed: int,
) -> ScenarioSpec:
    flows = tuple(SearchFlow)
    examples_per_flow = count // len(flows)
    flow = flows[index // examples_per_flow]
    example_index = index % examples_per_flow
    speech_styles = tuple(SpeechStyle)
    speech_style = speech_styles[example_index % len(speech_styles)]
    axis = SEARCH_SCENARIO_AXES[example_index % len(SEARCH_SCENARIO_AXES)]
    leakage_group_id = f"{flow.value}-{axis.topic.replace(' ', '-')}"
    return ScenarioSpec(
        scenario_id=f"scenario-{random_seed}-{index:06d}",
        random_seed=scenario_seed,
        family=flow.value,
        topic=axis.topic,
        length_band=LengthBand.MEDIUM,
        speech_style=speech_style,
        turns=_search_calibration_turns(flow, axis),
        split=(
            DatasetSplit.TRAIN
            if speech_style in (SpeechStyle.CLEAN, SpeechStyle.ASR_FRAGMENT, SpeechStyle.REPAIR)
            else DatasetSplit.VALIDATION
            if speech_style is SpeechStyle.CASUAL
            else DatasetSplit.TEST
        ),
        leakage_group_id=leakage_group_id,
    )


def _search_calibration_turns(
    flow: SearchFlow,
    axis: SearchScenarioAxis,
) -> tuple[AssistantTurnPlan, ...]:
    match flow:
        case SearchFlow.DIRECT:
            return (
                AssistantTurnPlan(
                    user_instruction=(
                        f"Make this fully specified request naturally: {axis.direct_request}"
                    ),
                    tool_steps=(PlannedToolStep(tool_name=ToolName.SEARCH),),
                    follow_up_kind=FollowUpKind.NONE,
                    utterance_form=UtteranceForm.CONTEXT_FIRST,
                ),
                AssistantTurnPlan(
                    user_instruction=(
                        "React to the result and ask for one concise restatement or filtering "
                        "that is fully supported by the returned text. Do not require a new fact."
                    ),
                    tool_steps=(),
                    follow_up_kind=FollowUpKind.CONSTRAINT,
                    utterance_form=UtteranceForm.REACTION,
                ),
            )
        case SearchFlow.CLARIFY_THEN_SEARCH:
            return (
                AssistantTurnPlan(
                    user_instruction=(
                        "Make this deliberately incomplete lookup request naturally: "
                        f"{axis.ambiguous_request}"
                    ),
                    tool_steps=(),
                    follow_up_kind=FollowUpKind.NONE,
                    utterance_form=UtteranceForm.CONTEXT_FIRST,
                    response_mode=AssistantResponseMode.CLARIFY_SEARCH,
                ),
                AssistantTurnPlan(
                    user_instruction=(
                        "Answer the assistant's clarification naturally. "
                        f"{axis.clarification} Keep the lookup request active."
                    ),
                    tool_steps=(),
                    follow_up_kind=FollowUpKind.CLARIFICATION,
                    utterance_form=UtteranceForm.FRAGMENT,
                    response_mode=AssistantResponseMode.CONFIRM_SEARCH,
                ),
                AssistantTurnPlan(
                    user_instruction=(
                        "Briefly confirm that the assistant should perform the proposed lookup. "
                        "Do not add or change any constraint."
                    ),
                    tool_steps=(PlannedToolStep(tool_name=ToolName.SEARCH),),
                    follow_up_kind=FollowUpKind.DELAYED_REQUEST,
                    utterance_form=UtteranceForm.REACTION,
                ),
            )
        case SearchFlow.REFINE_THEN_SEARCH:
            return (
                AssistantTurnPlan(
                    user_instruction=(
                        f"Make this fully specified request naturally: {axis.direct_request}"
                    ),
                    tool_steps=(PlannedToolStep(tool_name=ToolName.SEARCH),),
                    follow_up_kind=FollowUpKind.NONE,
                    utterance_form=UtteranceForm.REQUEST,
                ),
                AssistantTurnPlan(
                    user_instruction=(f"Make this correction naturally: {axis.correction}"),
                    tool_steps=(PlannedToolStep(tool_name=ToolName.SEARCH),),
                    follow_up_kind=FollowUpKind.CORRECTION,
                    utterance_form=UtteranceForm.SELF_REPAIR,
                ),
                AssistantTurnPlan(
                    user_instruction=(
                        "React to the corrected result with a short grounded acknowledgement or "
                        "request an exact restatement already supported by it."
                    ),
                    tool_steps=(),
                    follow_up_kind=FollowUpKind.PRONOUN,
                    utterance_form=UtteranceForm.REACTION,
                ),
            )


def _sample_bucket_segment_scenario(
    index: int,
    count: int,
    random_seed: int,
    scenario_seed: int,
    generator: random.Random,
) -> ScenarioSpec:
    buckets = tuple(SegmentBucket)
    examples_per_bucket = count // len(buckets)
    bucket = buckets[index // examples_per_bucket]
    example_index = index % examples_per_bucket
    template = _bucket_template(bucket, generator)
    turn_count = generator.randrange(2, 5)
    turns = _build_bucket_turns(
        bucket=bucket,
        template=template,
        turn_count=turn_count,
        example_index=example_index,
        generator=generator,
    )
    topic_variant = generator.choice(TOPIC_VARIANTS)
    leakage_group_id = f"segment-{bucket.value}-{topic_variant}"
    speech_styles = tuple(SpeechStyle)
    speech_style = speech_styles[example_index % len(speech_styles)]
    return ScenarioSpec(
        scenario_id=f"scenario-{random_seed}-{index:06d}",
        random_seed=scenario_seed,
        family=bucket.value,
        topic=f"{template.topic}: {topic_variant}",
        length_band=LengthBand.MEDIUM if turn_count < 4 else LengthBand.LONG,
        speech_style=speech_style,
        turns=turns,
        split=(
            DatasetSplit.TRAIN
            if speech_style in (SpeechStyle.CLEAN, SpeechStyle.ASR_FRAGMENT, SpeechStyle.REPAIR)
            else DatasetSplit.VALIDATION
            if speech_style is SpeechStyle.CASUAL
            else DatasetSplit.TEST
        ),
        leakage_group_id=leakage_group_id,
    )


def _bucket_template(
    bucket: SegmentBucket,
    generator: random.Random,
) -> ScenarioTemplate:
    match bucket:
        case SegmentBucket.NATURAL_CONVERSATION:
            return NATURAL_CONVERSATION_TEMPLATE
        case SegmentBucket.DRAFTING:
            return NO_TOOL_TEMPLATES[0]
        case SegmentBucket.BRAINSTORM_DECISION:
            return generator.choice(NO_TOOL_TEMPLATES[1:3])
        case SegmentBucket.SEARCH | SegmentBucket.CORRECTION_RECOVERY:
            return generator.choice(ONE_TOOL_TEMPLATES[:2])
        case SegmentBucket.CALCULATE:
            return ONE_TOOL_TEMPLATES[2]
        case SegmentBucket.GET_TIME:
            return ONE_TOOL_TEMPLATES[3]
        case SegmentBucket.SEQUENTIAL_TOOLS:
            return generator.choice(TWO_TOOL_TEMPLATES)


def _build_bucket_turns(
    bucket: SegmentBucket,
    template: ScenarioTemplate,
    turn_count: int,
    example_index: int,
    generator: random.Random,
) -> tuple[AssistantTurnPlan, ...]:
    if bucket is SegmentBucket.CORRECTION_RECOVERY:
        return _build_recovery_bucket_turns(
            template=template,
            turn_count=turn_count,
            example_index=example_index,
            generator=generator,
        )
    turns = [
        AssistantTurnPlan(
            user_instruction=template.user_instruction,
            tool_steps=tuple(
                PlannedToolStep(tool_name=tool_name) for tool_name in template.tool_names
            ),
            follow_up_kind=FollowUpKind.NONE,
            utterance_form=_sample_tool_turn_form(
                template,
                generator,
                is_follow_up=False,
            ),
        )
    ]
    return tuple(
        turns
        + list(
            _bucket_follow_up_turns(
                template=template,
                count=turn_count - 1,
                generator=generator,
            )
        )
    )


def _build_recovery_bucket_turns(
    template: ScenarioTemplate,
    turn_count: int,
    example_index: int,
    generator: random.Random,
) -> tuple[AssistantTurnPlan, ...]:
    adverse_outcomes = (
        PlannedOutcome.EMPTY,
        PlannedOutcome.FAILURE,
        PlannedOutcome.TIMEOUT,
    )
    first_outcome = (
        adverse_outcomes[example_index % len(adverse_outcomes)]
        if example_index % 2
        else PlannedOutcome.SUCCESS
    )
    turns = [
        AssistantTurnPlan(
            user_instruction=template.user_instruction,
            tool_steps=(
                PlannedToolStep(
                    tool_name=ToolName.SEARCH,
                    outcome=first_outcome,
                ),
            ),
            follow_up_kind=FollowUpKind.NONE,
            utterance_form=_sample_tool_turn_form(
                template,
                generator,
                is_follow_up=False,
            ),
        ),
        AssistantTurnPlan(
            user_instruction=(
                "Naturally correct or narrow one important detail from the preceding request and "
                "ask for a revised lookup. The correction must clearly require one new search."
            ),
            tool_steps=(PlannedToolStep(tool_name=ToolName.SEARCH),),
            follow_up_kind=FollowUpKind.CORRECTION,
            utterance_form=UtteranceForm.SELF_REPAIR,
        ),
    ]
    if turn_count > 2:
        turns.extend(
            _bucket_follow_up_turns(
                template=template,
                count=turn_count - 2,
                generator=generator,
            )
        )
    return tuple(turns)


def _bucket_follow_up_turns(
    template: ScenarioTemplate,
    count: int,
    generator: random.Random,
) -> tuple[AssistantTurnPlan, ...]:
    follow_up_kinds = (
        FollowUpKind.PRONOUN,
        FollowUpKind.CORRECTION,
        FollowUpKind.CONSTRAINT,
        FollowUpKind.CLARIFICATION,
    )
    turns: list[AssistantTurnPlan] = []
    for _ in range(count):
        follow_up_kind = generator.choice(follow_up_kinds)
        turns.append(
            AssistantTurnPlan(
                user_instruction=_follow_up_instruction(
                    template.family,
                    follow_up_kind,
                    permits_new_information_request=False,
                ),
                tool_steps=(),
                follow_up_kind=follow_up_kind,
                utterance_form=_sample_follow_up_form(generator, follow_up_kind),
            )
        )
    return tuple(turns)


def _sample_long_mixed_scenario(
    index: int,
    random_seed: int,
    scenario_seed: int,
    generator: random.Random,
) -> ScenarioSpec:
    target_turn_count = generator.randrange(8, 13)
    segment_count = 3 if target_turn_count <= 9 else 4
    segment_lengths = _long_segment_lengths(target_turn_count, segment_count, generator)
    segment_templates: list[ScenarioTemplate] = []
    turns: list[AssistantTurnPlan] = []
    for segment_index, segment_length in enumerate(segment_lengths):
        if segment_index % 2 == 0:
            template = generator.choice(NO_TOOL_TEMPLATES[:-1])
        else:
            template = _sample_long_tool_template(generator)
        segment_templates.append(template)
        turns.extend(
            _build_long_segment(
                template=template,
                turn_count=segment_length,
                is_opening=segment_index == 0,
                generator=generator,
            )
        )

    segment_families = tuple(template.family for template in segment_templates)
    leakage_group_id = f"long-mixed-{'-'.join(segment_families)}"
    topic = "multi-topic conversation: " + ", ".join(
        template.topic for template in segment_templates
    )
    return ScenarioSpec(
        scenario_id=f"scenario-{random_seed}-{index:06d}",
        random_seed=scenario_seed,
        family="long_mixed",
        topic=topic,
        length_band=LengthBand.LONG,
        speech_style=_weighted_choice(generator, SPEECH_STYLE_WEIGHTS),
        turns=tuple(turns),
        split=_split_for_group(leakage_group_id),
        leakage_group_id=leakage_group_id,
    )


def _long_segment_lengths(
    target_turn_count: int,
    segment_count: int,
    generator: random.Random,
) -> tuple[int, ...]:
    lengths = [2] * segment_count
    for _ in range(target_turn_count - sum(lengths)):
        available_indexes = tuple(index for index, length in enumerate(lengths) if length < 4)
        selected_index = generator.choice(available_indexes)
        lengths[selected_index] += 1
    return tuple(lengths)


def _sample_long_tool_template(generator: random.Random) -> ScenarioTemplate:
    template_group = generator.choices(
        population=(ONE_TOOL_TEMPLATES, TWO_TOOL_TEMPLATES),
        weights=(0.70, 0.30),
        k=1,
    )[0]
    return generator.choice(template_group)


def _build_long_segment(
    template: ScenarioTemplate,
    turn_count: int,
    is_opening: bool,
    generator: random.Random,
) -> tuple[AssistantTurnPlan, ...]:
    opening_instruction = (
        template.user_instruction
        if is_opening
        else (
            "Shift naturally to a new topic, then make this complete request without assuming "
            f"hidden context: {template.user_instruction}"
        )
    )
    turns = [
        AssistantTurnPlan(
            user_instruction=opening_instruction,
            tool_steps=_planned_tool_steps(template.tool_names, generator),
            follow_up_kind=FollowUpKind.NONE,
            utterance_form=_sample_tool_turn_form(
                template,
                generator,
                is_follow_up=False,
            ),
        )
    ]
    follow_up_kinds = (
        FollowUpKind.PRONOUN,
        FollowUpKind.CORRECTION,
        FollowUpKind.CONSTRAINT,
        FollowUpKind.CLARIFICATION,
    )
    for _ in range(turn_count - 1):
        follow_up_kind = generator.choice(follow_up_kinds)
        turns.append(
            AssistantTurnPlan(
                user_instruction=_follow_up_instruction(
                    template.family,
                    follow_up_kind,
                    permits_new_information_request=False,
                ),
                tool_steps=(),
                follow_up_kind=follow_up_kind,
                utterance_form=_sample_follow_up_form(generator, follow_up_kind),
            )
        )
    return tuple(turns)


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
        population=(0, 1, 2),
        weights=(0.35, 0.47, 0.18),
        k=1,
    )[0]
    match round_band:
        case 0:
            return generator.choice(NO_TOOL_TEMPLATES)
        case 1:
            return generator.choice(ONE_TOOL_TEMPLATES)
        case 2:
            return generator.choice(TWO_TOOL_TEMPLATES)
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
    if template.family == "stable_knowledge":
        turn_count = 1
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
                    user_instruction=_pre_tool_instruction(template, turn_index),
                    tool_steps=(),
                    follow_up_kind=(
                        FollowUpKind.NONE if turn_index == 0 else generator.choice(follow_up_kinds)
                    ),
                    utterance_form=_sample_pre_tool_form(
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
                    utterance_form=_sample_tool_turn_form(
                        template,
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
        tool_steps = (
            (PlannedToolStep(tool_name=ToolName.SEARCH),) if correction_requires_search else ()
        )
        turns.append(
            AssistantTurnPlan(
                user_instruction=_follow_up_instruction(
                    template.family,
                    follow_up_kind,
                    permits_new_information_request=bool(tool_steps),
                ),
                tool_steps=tool_steps,
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
    if (
        not template.tool_names
        or ToolName.GET_TIME in template.tool_names
        or turn_count == 1
        or generator.random() >= 0.43
    ):
        return 0
    return generator.randrange(1, min(turn_count, 3))


def _sample_length_band(
    template: ScenarioTemplate,
    generator: random.Random,
) -> LengthBand:
    if template.family == "stable_knowledge":
        return LengthBand.SHORT
    if not template.tool_names:
        return generator.choices(
            population=(LengthBand.SHORT, LengthBand.MEDIUM),
            weights=(0.40, 0.60),
            k=1,
        )[0]
    return _weighted_choice(generator, LENGTH_BAND_WEIGHTS)


def _pre_tool_instruction(template: ScenarioTemplate, turn_index: int) -> str:
    tool_specific_boundary = _pre_tool_boundary(template.tool_names)
    if turn_index == 0:
        return (
            "Make a related statement about the user's situation, preference, or plan. Do not ask "
            "a question or request information yet. "
            f"{tool_specific_boundary}"
        )
    return (
        "React naturally with another statement that sets up the later request. Do not ask for an "
        f"answer yet. {tool_specific_boundary}"
    )


def _pre_tool_boundary(tool_names: tuple[ToolName, ...]) -> str:
    boundaries: list[str] = []
    if ToolName.SEARCH in tool_names:
        boundaries.append("Do not ask for a current, local, or externally verified fact")
    if ToolName.GET_TIME in tool_names:
        boundaries.append(
            "Do not ask what time or date it is, how much time remains, or state a supposed "
            "current time"
        )
    if ToolName.CALCULATE in tool_names:
        boundaries.append(
            "Do not provide a complete arithmetic problem or ask for a numeric result"
        )
    if not boundaries:
        return "Keep it answerable as ordinary conversation without a tool."
    return "; ".join(boundaries) + "."


def _sample_utterance_form(
    generator: random.Random,
    is_follow_up: bool,
) -> UtteranceForm:
    return _weighted_choice(
        generator,
        FOLLOW_UP_FORM_WEIGHTS if is_follow_up else OPENING_FORM_WEIGHTS,
    )


def _sample_tool_turn_form(
    template: ScenarioTemplate,
    generator: random.Random,
    is_follow_up: bool,
) -> UtteranceForm:
    if template.tool_names or is_follow_up:
        return _sample_utterance_form(generator, is_follow_up)
    return generator.choices(
        population=(
            UtteranceForm.DIRECT_QUESTION,
            UtteranceForm.REQUEST,
            UtteranceForm.CONTEXT_FIRST,
            UtteranceForm.SELF_REPAIR,
        ),
        weights=(0.10, 0.35, 0.40, 0.15),
        k=1,
    )[0]


def _sample_pre_tool_form(
    generator: random.Random,
    is_follow_up: bool,
) -> UtteranceForm:
    if is_follow_up:
        return generator.choices(
            population=(
                UtteranceForm.CONTEXT_FIRST,
                UtteranceForm.FRAGMENT,
                UtteranceForm.REACTION,
                UtteranceForm.SELF_REPAIR,
            ),
            weights=(0.30, 0.20, 0.35, 0.15),
            k=1,
        )[0]
    return generator.choices(
        population=(
            UtteranceForm.CONTEXT_FIRST,
            UtteranceForm.FRAGMENT,
            UtteranceForm.SELF_REPAIR,
        ),
        weights=(0.55, 0.20, 0.25),
        k=1,
    )[0]


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


def _follow_up_instruction(
    scenario_family: str,
    follow_up_kind: FollowUpKind,
    permits_new_information_request: bool,
) -> str:
    if not permits_new_information_request:
        match scenario_family:
            case "draft_revision":
                instruction = (
                    "Ask for one concrete revision to the draft just produced, such as changing "
                    "its tone, length, or one phrase. Make clear this is still text drafting, not "
                    "a request to send anything."
                )
            case "constrained_brainstorm":
                instruction = (
                    "React to the suggestions already given, then select one or refine one "
                    "creative constraint. Do not imply that a hidden list, queue, or action exists."
                )
            case "personal_decision":
                instruction = (
                    "React to the advice and add or correct one personal preference. Ask only for "
                    "an updated opinion based on details already supplied in the conversation."
                )
            case "natural_conversation":
                instruction = (
                    "Continue the same personal thread with a specific reaction, small added "
                    "detail, or gentle correction. Invite a conversational response without "
                    "turning it into a factual advice request."
                )
            case _:
                instruction = _generic_follow_up_instruction(follow_up_kind)
    else:
        instruction = _generic_follow_up_instruction(follow_up_kind)
    if permits_new_information_request:
        return instruction
    return (
        f"{instruction} Do not introduce a new fact, calculation, current-information need, or "
        "external action. Only react to or clarify information already in the public history."
    )


def _generic_follow_up_instruction(follow_up_kind: FollowUpKind) -> str:
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
