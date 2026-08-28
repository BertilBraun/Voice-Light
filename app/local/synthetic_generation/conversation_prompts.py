from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, TypeAdapter, field_validator, model_validator

from app.local.synthetic_generation.models import SyntheticModel

ConversationPromptSetId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
conversation_prompt_set_id_adapter = TypeAdapter(ConversationPromptSetId)
SemanticCondition = Literal[
    "completion",
    "hold",
    "non_floor_feedback",
    "response_floor_claim",
    "interruption_floor_claim",
]
FloorCondition = Literal[
    "completion",
    "hold",
    "response_floor_claim",
    "interruption_floor_claim",
]


class IndependentPrefixAmbiguity(SyntheticModel):
    kind: Literal["independent"] = "independent"


class MatchedPrefixAmbiguity(SyntheticModel):
    kind: Literal["matched"] = "matched"
    pair_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    outcome: Literal["completion", "continuation"]


PrefixAmbiguity = Annotated[
    IndependentPrefixAmbiguity | MatchedPrefixAmbiguity,
    Field(discriminator="kind"),
]


class TopicDomain(StrEnum):
    DAILY_LIFE = "daily_life"
    TECHNOLOGY = "technology"
    TRAVEL = "travel"
    FOOD = "food"
    WORK = "work"
    SCIENCE = "science"
    RELATIONSHIPS = "relationships"
    ENTERTAINMENT = "entertainment"
    HISTORY = "history"
    HEALTH_AND_FITNESS = "health_and_fitness"
    EDUCATION = "education"
    ARTS_AND_CRAFTS = "arts_and_crafts"


class SpeechAct(StrEnum):
    QUESTION = "question"
    ANSWER = "answer"
    EXPLANATION = "explanation"
    OPINION = "opinion"
    ANECDOTE = "anecdote"
    REQUEST = "request"
    CORRECTION = "correction"


class EnglishAccent(StrEnum):
    GENERAL_AMERICAN = "general_american"
    MID_ATLANTIC_AMERICAN = "mid_atlantic_american"
    SOUTHERN_AMERICAN = "southern_american"
    CANADIAN = "canadian"
    SOUTHERN_BRITISH = "southern_british"
    NORTHERN_BRITISH = "northern_british"
    IRISH = "irish"
    AUSTRALIAN = "australian"
    NEW_ZEALAND = "new_zealand"


class PerceivedAge(StrEnum):
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    OLDER_ADULT = "older_adult"


class VocalPitch(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VocalWeight(StrEnum):
    LIGHT = "light"
    MEDIUM = "medium"
    FULL = "full"


class SpeakingPace(StrEnum):
    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"


class Energy(StrEnum):
    SUBDUED = "subdued"
    CONVERSATIONAL = "conversational"
    ANIMATED = "animated"


class Affect(StrEnum):
    NEUTRAL = "neutral"
    WARM = "warm"
    ENGAGING = "engaging"
    MYSTERIOUS = "mysterious"
    THOUGHTFUL = "thoughtful"
    PLAYFUL = "playful"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    URGENT = "urgent"


class MicroBackchannel(StrEnum):
    YEAH = "yeah"
    YEP = "yep"
    RIGHT = "right"
    OKAY = "okay"
    SURE = "sure"
    MHM = "mhm"
    UH_HUH = "uh-huh"
    MM_HMM = "mm-hmm"


class UserTurnLength(StrEnum):
    BRIEF = "brief"
    NORMAL = "normal"
    EXTENDED = "extended"


class BaseUserVoice(SyntheticModel):
    perceived_age: PerceivedAge
    accent: EnglishAccent
    pitch: VocalPitch
    vocal_weight: VocalWeight


class SegmentDelivery(SyntheticModel):
    pace: SpeakingPace
    energy: Energy
    affect: Affect


class AssistantTurnPrompt(SyntheticModel):
    turn_id: str = Field(pattern=r"^assistant_[1-9][0-9]*$")
    sequence_index: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=400)
    speaking_rate_words_per_minute: int = Field(ge=135, le=215)
    punctuation_pause_seconds: float = Field(ge=0.0, le=2.0)
    duration_variation_fraction: float = Field(default=0.15, ge=0.0, le=0.25)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_english_text(value)


class UserPromptBase(SyntheticModel):
    unit_id: str = Field(pattern=r"^user_[1-9][0-9]*$")
    sequence_index: int = Field(ge=0)
    speech_act: SpeechAct
    delivery: SegmentDelivery


class FloorOwningUserPromptBase(UserPromptBase):
    pass


class CompletionUserPrompt(FloorOwningUserPromptBase):
    condition: Literal["completion"] = "completion"
    text: str = Field(min_length=1, max_length=1200)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_english_text(value)


class HoldUserPrompt(FloorOwningUserPromptBase):
    condition: Literal["hold"] = "hold"
    text: str = Field(min_length=1, max_length=1200)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_english_text(value)


class NonFloorFeedbackUserPrompt(SyntheticModel):
    unit_id: str = Field(pattern=r"^user_[1-9][0-9]*$")
    sequence_index: int = Field(ge=0)
    delivery: SegmentDelivery
    condition: Literal["non_floor_feedback"] = "non_floor_feedback"
    text: MicroBackchannel
    during_assistant_turn_id: str = Field(pattern=r"^assistant_[1-9][0-9]*$")


class ResponseFloorClaimUserPrompt(FloorOwningUserPromptBase):
    condition: Literal["response_floor_claim"] = "response_floor_claim"
    text: str = Field(min_length=1, max_length=1200)
    after_assistant_turn_id: str = Field(pattern=r"^assistant_[1-9][0-9]*$")
    response_latency_seconds: float = Field(ge=0.05, le=2.5)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_english_text(value)


class InterruptionFloorClaimUserPrompt(FloorOwningUserPromptBase):
    condition: Literal["interruption_floor_claim"] = "interruption_floor_claim"
    text: str = Field(min_length=1, max_length=1200)
    during_assistant_turn_id: str = Field(pattern=r"^assistant_[1-9][0-9]*$")
    assistant_yield_delay_seconds: float = Field(ge=0.08, le=0.6)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_english_text(value)


FloorOwningUserPrompt = (
    CompletionUserPrompt
    | HoldUserPrompt
    | ResponseFloorClaimUserPrompt
    | InterruptionFloorClaimUserPrompt
)
UserPrompt = Annotated[
    CompletionUserPrompt
    | HoldUserPrompt
    | NonFloorFeedbackUserPrompt
    | ResponseFloorClaimUserPrompt
    | InterruptionFloorClaimUserPrompt,
    Field(discriminator="condition"),
]


class EnglishConversationPromptPlanData(SyntheticModel):
    schema_version: Literal["voice-light-english-conversation-prompts-v1"] = (
        "voice-light-english-conversation-prompts-v1"
    )
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    seed: int = Field(ge=0)
    domain: TopicDomain
    topic: str = Field(min_length=1)
    target_duration_seconds: float = Field(ge=60.0, le=120.0)
    base_user_voice: BaseUserVoice
    voice_reference_text: str = Field(min_length=1)
    prefix_ambiguity: PrefixAmbiguity = IndependentPrefixAmbiguity()
    assistant_turns: tuple[AssistantTurnPrompt, ...]
    user_prompts: tuple[UserPrompt, ...] = Field(min_length=3, max_length=12)

    @field_validator("topic")
    @classmethod
    def validate_english_topic(cls, value: str) -> str:
        return _validate_english_text(value)

    @field_validator("voice_reference_text")
    @classmethod
    def validate_voice_reference_text(cls, value: str) -> str:
        return _validate_english_text(value)


class EnglishConversationContentDraft(SyntheticModel):
    topic: str = Field(min_length=1)
    voice_reference_text: str = Field(min_length=1)
    opening_user_turn: str = Field(min_length=1, max_length=600)
    brief_user_turn: str = Field(min_length=1, max_length=160)
    normal_user_turn: str = Field(min_length=1, max_length=600)
    extended_user_turn: str = Field(min_length=1, max_length=1200)
    assistant_turns: tuple[str, str, str]

    @field_validator(
        "topic",
        "voice_reference_text",
        "opening_user_turn",
        "brief_user_turn",
        "normal_user_turn",
        "extended_user_turn",
    )
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_english_text(value)

    @field_validator("assistant_turns")
    @classmethod
    def validate_assistant_turns(cls, values: tuple[str, str, str]) -> tuple[str, str, str]:
        for value in values:
            _validate_english_text(value)
        return values


class EnglishConversationPromptPlan(EnglishConversationPromptPlanData):
    @model_validator(mode="after")
    def validate_semantic_plan(self) -> EnglishConversationPromptPlan:
        element_ids = tuple(turn.turn_id for turn in self.assistant_turns) + tuple(
            prompt.unit_id for prompt in self.user_prompts
        )
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("Conversation element IDs must be unique.")
        sequence_indices = tuple(turn.sequence_index for turn in self.assistant_turns) + tuple(
            prompt.sequence_index for prompt in self.user_prompts
        )
        if len(sequence_indices) != len(set(sequence_indices)):
            raise ValueError("Conversation sequence indices must be unique.")
        ordered_elements = sorted(
            (*self.assistant_turns, *self.user_prompts),
            key=lambda element: element.sequence_index,
        )
        if not ordered_elements or not _owns_user_floor(ordered_elements[0]):
            raise ValueError("A conversation must begin with a floor-owning user turn.")
        for element_index, element in enumerate(ordered_elements):
            match element:
                case AssistantTurnPrompt() if not _owns_user_floor(
                    ordered_elements[element_index - 1]
                ):
                    raise ValueError("Every assistant turn must directly reply to a user turn.")
                case _:
                    pass
        assistant_ids = {turn.turn_id for turn in self.assistant_turns}
        for prompt in self.user_prompts:
            match prompt:
                case NonFloorFeedbackUserPrompt(during_assistant_turn_id=turn_id):
                    _require_assistant_reference(turn_id, assistant_ids, prompt.unit_id)
                case ResponseFloorClaimUserPrompt(after_assistant_turn_id=turn_id):
                    _require_assistant_reference(turn_id, assistant_ids, prompt.unit_id)
                case InterruptionFloorClaimUserPrompt(during_assistant_turn_id=turn_id):
                    _require_assistant_reference(turn_id, assistant_ids, prompt.unit_id)
                case CompletionUserPrompt() | HoldUserPrompt():
                    pass
        return self


def assemble_conversation_prompt_plan(
    draft: EnglishConversationContentDraft,
    brief: ConversationGenerationBrief,
) -> EnglishConversationPromptPlan:
    generator = random.Random(brief.seed)
    base_voice = BaseUserVoice(
        perceived_age=generator.choice(tuple(PerceivedAge)),
        accent=generator.choice(tuple(EnglishAccent)),
        pitch=generator.choice(tuple(VocalPitch)),
        vocal_weight=generator.choice(tuple(VocalWeight)),
    )
    floor_turns = _ordered_floor_turns(draft, brief)
    assistant_turns: list[AssistantTurnPrompt] = []
    user_prompts: list[UserPrompt] = []
    sequence_index = 0
    user_index = 1
    user_prompts.append(
        CompletionUserPrompt(
            unit_id=f"user_{user_index}",
            sequence_index=sequence_index,
            speech_act=SpeechAct.OPINION,
            delivery=_delivery_for_index(brief, generator, 0),
            text=draft.opening_user_turn,
        )
    )
    sequence_index += 1
    user_index += 1
    include_feedback = "non_floor_feedback" in brief.required_conditions
    for assistant_index, (assistant_text, floor_turn) in enumerate(
        zip(draft.assistant_turns, floor_turns, strict=True),
        start=1,
    ):
        assistant_turn_id = f"assistant_{assistant_index}"
        assistant_turns.append(
            AssistantTurnPrompt(
                turn_id=assistant_turn_id,
                sequence_index=sequence_index,
                text=assistant_text,
                speaking_rate_words_per_minute=155 + assistant_index * 10,
                punctuation_pause_seconds=0.15 + assistant_index * 0.05,
                duration_variation_fraction=0.15,
            )
        )
        sequence_index += 1
        if include_feedback and assistant_index == 1:
            user_prompts.append(
                NonFloorFeedbackUserPrompt(
                    unit_id=f"user_{user_index}",
                    sequence_index=sequence_index,
                    delivery=_delivery_for_index(brief, generator, user_index),
                    text=generator.choice(tuple(MicroBackchannel)),
                    during_assistant_turn_id=assistant_turn_id,
                )
            )
            sequence_index += 1
            user_index += 1
        user_prompts.append(
            _floor_prompt(
                unit_id=f"user_{user_index}",
                sequence_index=sequence_index,
                text=floor_turn.text,
                condition=floor_turn.condition,
                assistant_turn_id=assistant_turn_id,
                speech_act=floor_turn.speech_act,
                delivery=_delivery_for_index(brief, generator, user_index),
                generator=generator,
            )
        )
        sequence_index += 1
        user_index += 1
    return EnglishConversationPromptPlan(
        plan_id=brief.plan_id,
        seed=brief.seed,
        domain=brief.domain,
        topic=draft.topic,
        target_duration_seconds=_target_duration_seconds(draft, brief.pace),
        base_user_voice=base_voice,
        voice_reference_text=_clone_ready_reference_text(draft.voice_reference_text),
        prefix_ambiguity=brief.prefix_ambiguity,
        assistant_turns=tuple(assistant_turns),
        user_prompts=tuple(user_prompts),
    )


def _owns_user_floor(element: AssistantTurnPrompt | UserPrompt) -> bool:
    match element:
        case FloorOwningUserPromptBase():
            return True
        case AssistantTurnPrompt() | NonFloorFeedbackUserPrompt():
            return False


def _clone_ready_reference_text(reference_text: str) -> str:
    return reference_text


class ConversationPromptGeneratorProvenance(SyntheticModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    requested_plan_count: int = Field(ge=5, le=2_000)
    target_conversation_hours: float | None = Field(default=None, gt=0.0, le=48.0)


class EnglishConversationPromptSet(SyntheticModel):
    schema_version: Literal["voice-light-english-conversation-prompt-set-v1"] = (
        "voice-light-english-conversation-prompt-set-v1"
    )
    set_id: ConversationPromptSetId
    provenance: ConversationPromptGeneratorProvenance
    plans: tuple[EnglishConversationPromptPlan, ...] = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_representative_set(self) -> EnglishConversationPromptSet:
        plan_ids = tuple(plan.plan_id for plan in self.plans)
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("Conversation plan IDs must be unique.")
        if len(self.plans) > self.provenance.requested_plan_count:
            raise ValueError("Prompt set contains more plans than its requested plan count.")
        target_hours = self.provenance.target_conversation_hours
        if target_hours is not None and len(self.plans) == self.provenance.requested_plan_count:
            planned_hours = sum(plan.target_duration_seconds for plan in self.plans) / 3600.0
            if planned_hours < target_hours:
                raise ValueError("Prompt set exhausted its plan limit before its duration target.")
        return self


class ConversationGenerationBrief(SyntheticModel):
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    seed: int = Field(ge=0)
    domain: TopicDomain
    pace: SpeakingPace
    affect: Affect
    required_conditions: tuple[SemanticCondition, ...] = Field(min_length=2)
    prefix_ambiguity: PrefixAmbiguity = IndependentPrefixAmbiguity()


@dataclass(frozen=True)
class _FloorTurn:
    text: str
    condition: FloorCondition
    speech_act: SpeechAct


def representative_conversation_briefs(
    count: int,
    seed: int,
) -> tuple[ConversationGenerationBrief, ...]:
    if not 5 <= count <= 2_000:
        raise ValueError("Conversation generation requires 5 to 2,000 conversations.")
    domains = tuple(TopicDomain)
    paces = tuple(SpeakingPace)
    affects = tuple(Affect)
    conditions = (
        "completion",
        "hold",
        "non_floor_feedback",
        "response_floor_claim",
        "interruption_floor_claim",
    )
    generator = random.Random(seed)
    domain_offset = generator.randrange(len(domains))
    pace_offset = generator.randrange(len(paces))
    affect_offset = generator.randrange(len(affects))
    briefs = []
    for plan_index in range(count):
        ambiguity_group_index = plan_index // 10
        ambiguity_group_offset = plan_index % 10
        include_matched_pair = count >= 10 and ambiguity_group_offset < 2
        ambiguity_digest = hashlib.sha256(
            f"{seed}:matched:{ambiguity_group_index}".encode()
        ).hexdigest()[:10]
        ambiguity_pair_id = f"ambiguity_{ambiguity_digest}"
        dimension_index = (
            ambiguity_group_index * 9 + max(0, ambiguity_group_offset - 1)
            if count >= 10
            else plan_index
        )
        first_condition = conditions[plan_index % len(conditions)]
        second_condition = conditions[(plan_index + 1) % len(conditions)]
        digest = hashlib.sha256(f"{seed}:{plan_index}".encode()).hexdigest()[:10]
        briefs.append(
            ConversationGenerationBrief(
                plan_id=f"pilot_{plan_index + 1:02d}_{digest}",
                seed=seed + plan_index,
                domain=domains[(domain_offset + plan_index) % len(domains)],
                pace=paces[(pace_offset + dimension_index) % len(paces)],
                affect=affects[(affect_offset + dimension_index) % len(affects)],
                required_conditions=(first_condition, second_condition),
                prefix_ambiguity=(
                    MatchedPrefixAmbiguity(
                        pair_id=ambiguity_pair_id,
                        outcome="completion" if plan_index == 0 else "continuation",
                    )
                    if include_matched_pair
                    else IndependentPrefixAmbiguity()
                ),
            )
        )
    return tuple(briefs)


def validate_conversation_prompt_set_id(value: str) -> str:
    return conversation_prompt_set_id_adapter.validate_python(value)


def conversation_generation_instruction(brief: ConversationGenerationBrief) -> str:
    brief_condition, normal_condition, extended_condition = _floor_conditions(brief)
    ordered_fields = _ordered_floor_turn_field_names(brief)
    json_schema = json.dumps(
        EnglishConversationContentDraft.model_json_schema(),
        separators=(",", ":"),
    )
    return f"""Write the natural English content for one coherent synthetic conversation.
Return only one compact JSON object accepted by the schema at the end of this instruction. Code
will add all IDs, interaction conditions, references, timing, voice metadata, and delivery settings.
Do not add any of those structural fields.

Fixed requirements:
- The domain is {brief.domain.value!r}. This item has the unique scenario key {brief.plan_id!r}.
- Ground the topic and exchange in at least two of these independent scenario cues:
  {_scenario_cues(brief)}. Interpret them freely and naturally; they are inspiration rather than
  words that must be repeated literally.
- Invent a concrete situation specific to this item. Do not default to recurring corpus themes
  such as quantum computing, classroom seating, remote software teams, sustainable seafood,
  farming, or generic productivity advice unless the scenario cues make one genuinely relevant.
- Use this initial conversational direction: {_creative_direction(brief)}
- The user always opens the conversation. The assistant never initiates a conversation or starts a
  new turn without directly replying to a floor-owning user turn.
- The fields form this chronological exchange: opening_user_turn, assistant_turns[0],
  {ordered_fields[0]}, assistant_turns[1], {ordered_fields[1]}, assistant_turns[2],
  {ordered_fields[2]}.
  Make every response directly acknowledge and develop the preceding text so the exchange reads
  naturally when interleaved in that exact order.
- The code will use brief_user_turn as {_condition_content_instruction(brief_condition)}
- The code will use normal_user_turn as {_condition_content_instruction(normal_condition)}
- The code will use extended_user_turn as {_condition_content_instruction(extended_condition)}
{_prefix_ambiguity_instruction(brief.prefix_ambiguity)}
- Vary the turn lengths naturally. Aim for brief_user_turn to be a word or a short sentence,
  normal_user_turn to be a typical conversational response, and extended_user_turn to contain a
  longer train of thought that could occupy roughly 15-30 seconds. These are creative directions,
  not exact word-count requirements. Keep assistant replies conversational rather than uniformly
  terse.
- Write voice_reference_text as one neutral English sentence suitable for a clean, short reference
  render. It need not mention the conversation topic.
- All text is natural modern English. Do not include stage
  directions, sound effects, copyrighted passages, real public figures, or unsafe content.
- Do not output backchannels. Code inserts a strict one-token acknowledgement when required.
- Do not output plans, IDs, sequence numbers, semantic labels, durations, pauses, numeric
  parameters, accents, voice descriptions, pace, affect, or any other metadata.

JSON schema:
{json_schema}
"""


def _creative_direction(brief: ConversationGenerationBrief) -> str:
    directions = (
        "make a practical decision after weighing imperfect options",
        "work through a minor disagreement without sounding theatrical",
        "recount a recent experience and revise an initial assumption",
        "ask for clarification while already knowing part of the answer",
        "coordinate concrete next steps under mild time pressure",
        "compare two plausible approaches using personal observations",
        "explain an unexpected result and explore what caused it",
        "seek advice about an ordinary situation with a subtle complication",
        "troubleshoot a problem through a natural back-and-forth",
        "share an opinion that becomes more nuanced during the exchange",
        "plan something enjoyable while negotiating preferences",
        "reflect on a small change that had an unintended consequence",
    )
    return directions[brief.seed % len(directions)]


def _scenario_cues(brief: ConversationGenerationBrief) -> str:
    settings = (
        "an apartment lobby",
        "a neighborhood repair shop",
        "a museum archive",
        "a ferry terminal",
        "a rehearsal studio",
        "a community clinic",
        "a university lab",
        "a crowded café",
        "a public library",
        "a small hotel",
        "a makerspace",
        "a sports center",
        "a local council office",
        "a train platform",
        "a family kitchen",
        "an outdoor market",
        "a shared workshop",
        "a school theater",
        "a wildlife center",
        "a recording booth",
        "a rooftop garden",
        "a coastal visitor center",
        "a volunteer meeting",
        "a home office",
        "a bookshop",
        "a bicycle garage",
        "a conference hallway",
        "a music venue",
        "a pottery studio",
        "a neighborhood park",
        "a language class",
    )
    complications = (
        "a missing receipt",
        "an ambiguous instruction",
        "a last-minute cancellation",
        "an unexpected repair",
        "a confusing measurement",
        "a double booking",
        "a delayed delivery",
        "a misunderstood message",
        "a limited budget",
        "a changing weather forecast",
        "a forgotten commitment",
        "a surprising test result",
        "an inaccessible entrance",
        "a damaged borrowed item",
        "a scheduling conflict",
        "a disputed memory",
        "an unfamiliar rule",
        "a noisy environment",
        "a missing ingredient",
        "a reluctant participant",
        "a software update",
        "a misplaced key",
        "a sensitive deadline",
        "an unclear price",
        "a minor injury",
        "an incorrect assumption",
        "a crowded waiting list",
        "a broken promise",
        "a change of ownership",
    )
    perspectives = (
        "a first-time visitor",
        "an experienced volunteer",
        "a skeptical colleague",
        "a careful beginner",
        "a tired parent",
        "an enthusiastic neighbor",
        "a returning customer",
        "a new team member",
        "a practical hobbyist",
        "a concerned friend",
        "an independent contractor",
        "a curious student",
        "a cautious organizer",
        "a regular commuter",
        "a recent graduate",
        "a longtime resident",
        "a visiting relative",
        "a shift supervisor",
        "a club member",
        "a small business owner",
        "a patient instructor",
        "a reluctant guest",
        "an observant bystander",
    )
    digest = hashlib.sha256(f"scenario:{brief.plan_id}:{brief.seed}".encode()).digest()
    return ", ".join(
        (
            settings[int.from_bytes(digest[0:4], "big") % len(settings)],
            complications[int.from_bytes(digest[4:8], "big") % len(complications)],
            perspectives[int.from_bytes(digest[8:12], "big") % len(perspectives)],
        )
    )


def qwen_voice_instruction(
    base_voice: BaseUserVoice,
    delivery: SegmentDelivery,
) -> str:
    return (
        f"A {base_voice.perceived_age.value.replace('_', ' ')} English speaker with a "
        f"{base_voice.accent.value.replace('_', ' ')} accent, {base_voice.pitch.value} pitch, "
        f"and {base_voice.vocal_weight.value} vocal weight. Speak at a {delivery.pace.value} "
        f"pace with {delivery.energy.value} energy and a {delivery.affect.value} affect. "
        "Use natural voiced projection in a clean, close-mic studio recording."
    )


def _delivery_for_index(
    brief: ConversationGenerationBrief,
    generator: random.Random,
    index: int,
) -> SegmentDelivery:
    if index == 0:
        return SegmentDelivery(
            pace=brief.pace,
            energy=Energy.ANIMATED if brief.pace is SpeakingPace.FAST else Energy.CONVERSATIONAL,
            affect=brief.affect,
        )
    return SegmentDelivery(
        pace=generator.choice(tuple(SpeakingPace)),
        energy=generator.choice(tuple(Energy)),
        affect=generator.choice(tuple(Affect)),
    )


def _floor_prompt(
    unit_id: str,
    sequence_index: int,
    text: str,
    condition: FloorCondition,
    assistant_turn_id: str,
    speech_act: SpeechAct,
    delivery: SegmentDelivery,
    generator: random.Random,
) -> FloorOwningUserPrompt:
    match condition:
        case "completion":
            return CompletionUserPrompt(
                unit_id=unit_id,
                sequence_index=sequence_index,
                speech_act=speech_act,
                delivery=delivery,
                text=text,
            )
        case "hold":
            return HoldUserPrompt(
                unit_id=unit_id,
                sequence_index=sequence_index,
                speech_act=speech_act,
                delivery=delivery,
                text=text,
            )
        case "response_floor_claim":
            return ResponseFloorClaimUserPrompt(
                unit_id=unit_id,
                sequence_index=sequence_index,
                speech_act=speech_act,
                delivery=delivery,
                text=text,
                after_assistant_turn_id=assistant_turn_id,
                response_latency_seconds=round(generator.uniform(0.05, 2.5), 3),
            )
        case "interruption_floor_claim":
            return InterruptionFloorClaimUserPrompt(
                unit_id=unit_id,
                sequence_index=sequence_index,
                speech_act=speech_act,
                delivery=delivery,
                text=text,
                during_assistant_turn_id=assistant_turn_id,
                assistant_yield_delay_seconds=round(generator.uniform(0.08, 0.6), 3),
            )
        case _:
            raise AssertionError(f"Unsupported floor condition: {condition}")


def _floor_conditions(brief: ConversationGenerationBrief) -> tuple[FloorCondition, ...]:
    required_floor_conditions: tuple[FloorCondition, ...] = tuple(
        condition
        for condition in brief.required_conditions
        if condition not in {"completion", "non_floor_feedback"}
    )
    match brief.prefix_ambiguity:
        case IndependentPrefixAmbiguity():
            return required_floor_conditions + ("completion",) * (
                3 - len(required_floor_conditions)
            )
        case MatchedPrefixAmbiguity(outcome=outcome):
            matched_condition: FloorCondition = (
                "hold" if outcome == "continuation" else "completion"
            )
            remaining = tuple(
                condition
                for condition in required_floor_conditions
                if condition != matched_condition
            )
            return (
                remaining[0] if remaining else "completion",
                matched_condition,
                remaining[1] if len(remaining) > 1 else "completion",
            )


def _ordered_floor_turns(
    draft: EnglishConversationContentDraft,
    brief: ConversationGenerationBrief,
) -> tuple[_FloorTurn, ...]:
    conditions = _floor_conditions(brief)
    turns_by_field = {
        "brief_user_turn": _FloorTurn(
            text=draft.brief_user_turn,
            condition=conditions[0],
            speech_act=SpeechAct.ANSWER,
        ),
        "normal_user_turn": _FloorTurn(
            text=draft.normal_user_turn,
            condition=conditions[1],
            speech_act=SpeechAct.EXPLANATION,
        ),
        "extended_user_turn": _FloorTurn(
            text=draft.extended_user_turn,
            condition=conditions[2],
            speech_act=SpeechAct.ANECDOTE,
        ),
    }
    return tuple(
        turns_by_field[field_name] for field_name in _ordered_floor_turn_field_names(brief)
    )


def _ordered_floor_turn_field_names(
    brief: ConversationGenerationBrief,
) -> tuple[Literal["brief_user_turn", "normal_user_turn", "extended_user_turn"], ...]:
    permutations = (
        ("brief_user_turn", "normal_user_turn", "extended_user_turn"),
        ("brief_user_turn", "extended_user_turn", "normal_user_turn"),
        ("normal_user_turn", "brief_user_turn", "extended_user_turn"),
        ("normal_user_turn", "extended_user_turn", "brief_user_turn"),
        ("extended_user_turn", "brief_user_turn", "normal_user_turn"),
        ("extended_user_turn", "normal_user_turn", "brief_user_turn"),
    )
    return permutations[brief.seed % len(permutations)]


def _prefix_ambiguity_instruction(prefix_ambiguity: PrefixAmbiguity) -> str:
    match prefix_ambiguity:
        case IndependentPrefixAmbiguity():
            return ""
        case MatchedPrefixAmbiguity(outcome="completion"):
            return (
                "- normal_user_turn is the completion member of a distribution-matched ambiguity "
                "pair. Give it a natural 5-12 word opening clause that could plausibly continue, "
                "then let the turn reach a genuine stopping point. Do not use stock phrases or "
                "mention the pairing."
            )
        case MatchedPrefixAmbiguity(outcome="continuation"):
            return (
                "- normal_user_turn is the continuation member of a distribution-matched "
                "ambiguity pair. Give it a natural 5-12 word opening clause that could plausibly "
                "stop, then continue the same thought without a written pause marker. Do not use "
                "stock phrases or mention the pairing."
            )


def _condition_content_instruction(condition: FloorCondition) -> str:
    match condition:
        case "completion":
            return "a complete floor-owning turn that reaches a natural stopping point."
        case "hold":
            return (
                "one continuous floor-owning thought that naturally invites a hesitation in its "
                "spoken delivery; do not add a written pause or split the sentence."
            )
        case "response_floor_claim":
            return "a direct floor-owning answer to the immediately preceding assistant turn."
        case "interruption_floor_claim":
            return (
                "a clear, assertive floor claim that can begin before the immediately preceding "
                "assistant turn finishes."
            )


def _target_duration_seconds(
    draft: EnglishConversationContentDraft,
    pace: SpeakingPace,
) -> float:
    user_words = sum(
        len(text.split())
        for text in (
            draft.opening_user_turn,
            draft.brief_user_turn,
            draft.normal_user_turn,
            draft.extended_user_turn,
        )
    )
    assistant_words = sum(len(text.split()) for text in draft.assistant_turns)
    user_words_per_minute = {
        SpeakingPace.SLOW: 125,
        SpeakingPace.MODERATE: 165,
        SpeakingPace.FAST: 205,
    }[pace]
    estimated_seconds = user_words * 60 / user_words_per_minute + assistant_words * 60 / 175 + 15
    return round(min(120.0, max(60.0, estimated_seconds)), 3)


def _validate_english_text(value: str) -> str:
    if any(not character.isprintable() for character in value):
        raise ValueError("English prompt text must use printable text.")
    if not any(character.isalnum() for character in value):
        raise ValueError("English prompt text must contain alphanumeric text.")
    return value


def _user_turn_length(texts: tuple[str, ...]) -> UserTurnLength:
    word_count = sum(len(text.split()) for text in texts)
    if word_count <= 11:
        return UserTurnLength.BRIEF
    if word_count <= 44:
        return UserTurnLength.NORMAL
    return UserTurnLength.EXTENDED


def _require_assistant_reference(
    turn_id: str,
    assistant_ids: set[str],
    unit_id: str,
) -> None:
    if turn_id not in assistant_ids:
        raise ValueError(f"User prompt {unit_id} references unknown assistant turn {turn_id}.")


def floor_user_turn_length(prompt: UserPrompt) -> UserTurnLength | None:
    match prompt:
        case (
            CompletionUserPrompt()
            | HoldUserPrompt()
            | ResponseFloorClaimUserPrompt()
            | InterruptionFloorClaimUserPrompt()
        ):
            return _user_turn_length(_user_prompt_texts(prompt))
        case NonFloorFeedbackUserPrompt():
            return None


def _user_prompt_texts(prompt: UserPrompt) -> tuple[str, ...]:
    match prompt:
        case (
            CompletionUserPrompt(text=text)
            | HoldUserPrompt(text=text)
            | NonFloorFeedbackUserPrompt(text=text)
            | ResponseFloorClaimUserPrompt(text=text)
            | InterruptionFloorClaimUserPrompt(text=text)
        ):
            return (text,)
