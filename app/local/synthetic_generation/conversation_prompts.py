from __future__ import annotations

import hashlib
import json
import random
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

    @model_validator(mode="after")
    def validate_turn_length(self) -> CompletionUserPrompt:
        _user_turn_length((self.text,))
        return self


class HoldUserPrompt(FloorOwningUserPromptBase):
    condition: Literal["hold"] = "hold"
    text_before_pause: str = Field(min_length=1, max_length=350)
    text_after_pause: str = Field(min_length=1, max_length=350)
    pause_duration_seconds: float = Field(ge=0.5, le=2.5)

    @field_validator("text_before_pause", "text_after_pause")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_english_text(value)

    @model_validator(mode="after")
    def validate_turn_length(self) -> HoldUserPrompt:
        _user_turn_length((self.text_before_pause, self.text_after_pause))
        return self


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

    @model_validator(mode="after")
    def validate_turn_length(self) -> ResponseFloorClaimUserPrompt:
        _user_turn_length((self.text,))
        return self


class InterruptionFloorClaimUserPrompt(FloorOwningUserPromptBase):
    condition: Literal["interruption_floor_claim"] = "interruption_floor_claim"
    text: str = Field(min_length=1, max_length=1200)
    during_assistant_turn_id: str = Field(pattern=r"^assistant_[1-9][0-9]*$")
    assistant_yield_delay_seconds: float = Field(ge=0.08, le=0.6)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_english_text(value)

    @model_validator(mode="after")
    def validate_turn_length(self) -> InterruptionFloorClaimUserPrompt:
        _user_turn_length((self.text,))
        return self


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
    topic: str = Field(min_length=1, max_length=120)
    target_duration_seconds: float = Field(ge=60.0, le=120.0)
    base_user_voice: BaseUserVoice
    voice_reference_text: str = Field(min_length=1, max_length=180)
    assistant_turns: tuple[AssistantTurnPrompt, ...]
    user_prompts: tuple[UserPrompt, ...] = Field(min_length=3, max_length=12)

    @field_validator("topic")
    @classmethod
    def validate_english_topic(cls, value: str) -> str:
        return _validate_english_text(value)

    @field_validator("voice_reference_text")
    @classmethod
    def validate_voice_reference_text(cls, value: str) -> str:
        _validate_english_text(value)
        word_count = len(value.split())
        if not 8 <= word_count <= 18:
            raise ValueError("Voice reference text requires 8 to 18 words.")
        return value


class EnglishConversationContentDraft(SyntheticModel):
    topic: str = Field(min_length=1, max_length=120)
    voice_reference_text: str = Field(min_length=1, max_length=180)
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
            _validate_word_count("Assistant turn", value, 3, 25)
        return values

    @model_validator(mode="after")
    def validate_content_lengths(self) -> EnglishConversationContentDraft:
        _validate_word_count("Voice reference text", self.voice_reference_text, 8, 18)
        _validate_word_count("Opening user turn", self.opening_user_turn, 12, 44)
        _validate_word_count("Brief user turn", self.brief_user_turn, 2, 11)
        _validate_word_count("Normal user turn", self.normal_user_turn, 12, 44)
        _validate_word_count("Extended user turn", self.extended_user_turn, 45, 90)
        return self


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
        assistant_ids = {turn.turn_id for turn in self.assistant_turns}
        realized_length_bands = {
            length_band
            for prompt in self.user_prompts
            if (length_band := floor_user_turn_length(prompt)) is not None
        }
        if realized_length_bands != set(UserTurnLength):
            raise ValueError("A conversation requires brief, normal, and extended user turns.")
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
    floor_conditions = _floor_conditions(brief)
    floor_texts = (draft.brief_user_turn, draft.normal_user_turn, draft.extended_user_turn)
    speech_acts = (SpeechAct.ANSWER, SpeechAct.EXPLANATION, SpeechAct.ANECDOTE)
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
    for assistant_index, (assistant_text, user_text, condition, speech_act) in enumerate(
        zip(draft.assistant_turns, floor_texts, floor_conditions, speech_acts, strict=True),
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
                text=user_text,
                condition=condition,
                assistant_turn_id=assistant_turn_id,
                speech_act=speech_act,
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
        voice_reference_text=draft.voice_reference_text,
        assistant_turns=tuple(assistant_turns),
        user_prompts=tuple(user_prompts),
    )


class ConversationPromptGeneratorProvenance(SyntheticModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    requested_plan_count: int = Field(ge=5, le=10)


class EnglishConversationPromptSet(SyntheticModel):
    schema_version: Literal["voice-light-english-conversation-prompt-set-v1"] = (
        "voice-light-english-conversation-prompt-set-v1"
    )
    set_id: ConversationPromptSetId
    provenance: ConversationPromptGeneratorProvenance
    plans: tuple[EnglishConversationPromptPlan, ...] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_representative_set(self) -> EnglishConversationPromptSet:
        plan_ids = tuple(plan.plan_id for plan in self.plans)
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("Conversation plan IDs must be unique.")
        if len(self.plans) > self.provenance.requested_plan_count:
            raise ValueError("Prompt set contains more plans than its requested plan count.")
        reference_texts = tuple(
            " ".join(plan.voice_reference_text.casefold().split()) for plan in self.plans
        )
        if len(reference_texts) != len(set(reference_texts)):
            raise ValueError("Voice reference texts must be unique across a prompt set.")
        normalized_texts = tuple(
            " ".join(text.casefold().split())
            for plan in self.plans
            for prompt in plan.user_prompts
            for text in user_prompt_texts_for_uniqueness(prompt)
        )
        if len(normalized_texts) != len(set(normalized_texts)):
            raise ValueError("User prompt texts must be unique across a prompt set.")
        return self


class ConversationGenerationBrief(SyntheticModel):
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    seed: int = Field(ge=0)
    domain: TopicDomain
    pace: SpeakingPace
    affect: Affect
    required_conditions: tuple[SemanticCondition, ...] = Field(min_length=2)


def representative_conversation_briefs(
    count: int,
    seed: int,
) -> tuple[ConversationGenerationBrief, ...]:
    if not 5 <= count <= 10:
        raise ValueError("A representative pilot requires 5 to 10 conversations.")
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
        first_condition = conditions[plan_index % len(conditions)]
        second_condition = conditions[(plan_index + 1) % len(conditions)]
        digest = hashlib.sha256(f"{seed}:{plan_index}".encode()).hexdigest()[:10]
        briefs.append(
            ConversationGenerationBrief(
                plan_id=f"pilot_{plan_index + 1:02d}_{digest}",
                seed=seed + plan_index,
                domain=domains[(domain_offset + plan_index) % len(domains)],
                pace=paces[(pace_offset + plan_index) % len(paces)],
                affect=affects[(affect_offset + plan_index) % len(affects)],
                required_conditions=(first_condition, second_condition),
            )
        )
    return tuple(briefs)


def validate_conversation_prompt_set_id(value: str) -> str:
    return conversation_prompt_set_id_adapter.validate_python(value)


def conversation_generation_instruction(brief: ConversationGenerationBrief) -> str:
    brief_condition, normal_condition, extended_condition = _floor_conditions(brief)
    json_schema = json.dumps(
        EnglishConversationContentDraft.model_json_schema(),
        separators=(",", ":"),
    )
    return f"""Write the natural English content for one coherent synthetic conversation.
Return only one compact JSON object accepted by the schema at the end of this instruction. Code
will add all IDs, interaction conditions, references, timing, voice metadata, and delivery settings.
Do not add any of those structural fields.

Fixed requirements:
- The domain is {brief.domain.value!r}; invent a specific topic unlike generic rural or farming
  stories.
- The fields form this chronological exchange: opening_user_turn, assistant_turns[0],
  brief_user_turn, assistant_turns[1], normal_user_turn, assistant_turns[2], extended_user_turn.
  Make every response directly acknowledge and develop the preceding text so the exchange reads
  naturally when interleaved in that exact order.
- The code will use brief_user_turn as {_condition_content_instruction(brief_condition)}
- The code will use normal_user_turn as {_condition_content_instruction(normal_condition)}
- The code will use extended_user_turn as {_condition_content_instruction(extended_condition)}
- opening_user_turn and normal_user_turn each contain 12-44 words. brief_user_turn contains 2-11
  words. extended_user_turn contains 45-90 words and should sound like 20-30 seconds of natural
  speech. Each assistant turn contains 3-25 words.
- Write voice_reference_text as one exact, neutral English sentence of 8-18 words suitable for a
  clean 3-8 second reference render. It need not mention the conversation topic.
- All text is natural modern English with printable ASCII punctuation. Do not include stage
  directions, sound effects, copyrighted passages, real public figures, or unsafe content.
- Do not output backchannels. Code inserts a strict one-token acknowledgement when required.
- Do not output plans, IDs, sequence numbers, semantic labels, durations, pauses, numeric
  parameters, accents, voice descriptions, pace, affect, or any other metadata.

JSON schema:
{json_schema}
"""


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
            before, after = _split_hold_text(text)
            return HoldUserPrompt(
                unit_id=unit_id,
                sequence_index=sequence_index,
                speech_act=speech_act,
                delivery=delivery,
                text_before_pause=before,
                text_after_pause=after,
                pause_duration_seconds=round(generator.uniform(0.5, 2.5), 3),
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
    return required_floor_conditions + ("completion",) * (3 - len(required_floor_conditions))


def _condition_content_instruction(condition: FloorCondition) -> str:
    match condition:
        case "completion":
            return "a complete floor-owning turn that reaches a natural stopping point."
        case "hold":
            return (
                "one continuous floor-owning thought that remains grammatical and coherent when "
                "code inserts a hesitation pause near its midpoint; do not add a written pause."
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


def _split_hold_text(text: str) -> tuple[str, str]:
    words = text.split()
    split_index = len(words) // 2
    return " ".join(words[:split_index]), " ".join(words[split_index:])


def _validate_english_text(value: str) -> str:
    if any(not character.isprintable() for character in value):
        raise ValueError("English prompt text must use printable text.")
    alphabetic_characters = tuple(character for character in value if character.isalpha())
    if not alphabetic_characters or any(
        not character.isascii() for character in alphabetic_characters
    ):
        raise ValueError("English prompt text must use unaccented Latin letters.")
    return value


def _validate_word_count(label: str, text: str, minimum: int, maximum: int) -> None:
    word_count = len(text.split())
    if not minimum <= word_count <= maximum:
        raise ValueError(f"{label} requires {minimum} to {maximum} words; received {word_count}.")


def _user_turn_length(texts: tuple[str, ...]) -> UserTurnLength:
    word_count = sum(len(text.split()) for text in texts)
    if 2 <= word_count <= 11:
        return UserTurnLength.BRIEF
    if 12 <= word_count <= 44:
        return UserTurnLength.NORMAL
    if 45 <= word_count <= 90:
        return UserTurnLength.EXTENDED
    raise ValueError(
        f"Floor-owning user turns require 2-11, 12-44, or 45-90 words; received {word_count}."
    )


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


def user_prompt_texts_for_uniqueness(prompt: UserPrompt) -> tuple[str, ...]:
    match prompt:
        case NonFloorFeedbackUserPrompt():
            return ()
        case (
            CompletionUserPrompt()
            | HoldUserPrompt()
            | ResponseFloorClaimUserPrompt()
            | InterruptionFloorClaimUserPrompt()
        ):
            return _user_prompt_texts(prompt)


def _user_prompt_texts(prompt: UserPrompt) -> tuple[str, ...]:
    match prompt:
        case HoldUserPrompt(text_before_pause=before, text_after_pause=after):
            return (before, after)
        case (
            CompletionUserPrompt(text=text)
            | NonFloorFeedbackUserPrompt(text=text)
            | ResponseFloorClaimUserPrompt(text=text)
            | InterruptionFloorClaimUserPrompt(text=text)
        ):
            return (text,)
