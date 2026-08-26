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


class NonFloorFeedbackKind(StrEnum):
    BACKCHANNEL = "backchannel"
    REACTION = "reaction"
    COLLABORATIVE_COMPLETION = "collaborative_completion"


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


class CompletionUserPrompt(UserPromptBase):
    condition: Literal["completion"] = "completion"
    text: str = Field(min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_short_user_text(value)


class HoldUserPrompt(UserPromptBase):
    condition: Literal["hold"] = "hold"
    text_before_pause: str = Field(min_length=1, max_length=350)
    text_after_pause: str = Field(min_length=1, max_length=350)
    pause_duration_seconds: float = Field(ge=0.5, le=2.5)

    @field_validator("text_before_pause", "text_after_pause")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_short_user_text(value)


class NonFloorFeedbackUserPrompt(UserPromptBase):
    condition: Literal["non_floor_feedback"] = "non_floor_feedback"
    text: str = Field(min_length=1, max_length=80)
    feedback_kind: NonFloorFeedbackKind
    during_assistant_turn_id: str = Field(pattern=r"^assistant_[1-9][0-9]*$")

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_short_user_text(value, maximum_words=8)


class ResponseFloorClaimUserPrompt(UserPromptBase):
    condition: Literal["response_floor_claim"] = "response_floor_claim"
    text: str = Field(min_length=1, max_length=500)
    after_assistant_turn_id: str = Field(pattern=r"^assistant_[1-9][0-9]*$")
    response_latency_seconds: float = Field(ge=0.05, le=2.5)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_short_user_text(value)


class InterruptionFloorClaimUserPrompt(UserPromptBase):
    condition: Literal["interruption_floor_claim"] = "interruption_floor_claim"
    text: str = Field(min_length=1, max_length=500)
    during_assistant_turn_id: str = Field(pattern=r"^assistant_[1-9][0-9]*$")
    assistant_yield_delay_seconds: float = Field(ge=0.08, le=0.6)

    @field_validator("text")
    @classmethod
    def validate_english_text(cls, value: str) -> str:
        return _validate_short_user_text(value)


UserPrompt = Annotated[
    CompletionUserPrompt
    | HoldUserPrompt
    | NonFloorFeedbackUserPrompt
    | ResponseFloorClaimUserPrompt
    | InterruptionFloorClaimUserPrompt,
    Field(discriminator="condition"),
]


class EnglishConversationPromptPlan(SyntheticModel):
    schema_version: Literal["voice-light-english-conversation-prompts-v1"] = (
        "voice-light-english-conversation-prompts-v1"
    )
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    seed: int = Field(ge=0)
    domain: TopicDomain
    topic: str = Field(min_length=1, max_length=120)
    target_duration_seconds: float = Field(ge=20.0, le=30.0)
    base_user_voice: BaseUserVoice
    assistant_turns: tuple[AssistantTurnPrompt, ...]
    user_prompts: tuple[UserPrompt, ...] = Field(min_length=1, max_length=8)

    @field_validator("topic")
    @classmethod
    def validate_english_topic(cls, value: str) -> str:
        return _validate_english_text(value)

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


class ConversationPromptGeneratorProvenance(SyntheticModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    requested_plan_count: int = Field(ge=10, le=20)


class EnglishConversationPromptSet(SyntheticModel):
    schema_version: Literal["voice-light-english-conversation-prompt-set-v1"] = (
        "voice-light-english-conversation-prompt-set-v1"
    )
    set_id: ConversationPromptSetId
    provenance: ConversationPromptGeneratorProvenance
    plans: tuple[EnglishConversationPromptPlan, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_representative_set(self) -> EnglishConversationPromptSet:
        plan_ids = tuple(plan.plan_id for plan in self.plans)
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("Conversation plan IDs must be unique.")
        if len(self.plans) > self.provenance.requested_plan_count:
            raise ValueError("Prompt set contains more plans than its requested plan count.")
        normalized_texts = tuple(
            " ".join(text.casefold().split())
            for plan in self.plans
            for prompt in plan.user_prompts
            for text in _user_prompt_texts(prompt)
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
    required_conditions: tuple[
        Literal[
            "completion",
            "hold",
            "non_floor_feedback",
            "response_floor_claim",
            "interruption_floor_claim",
        ],
        ...,
    ] = Field(min_length=2)


def representative_conversation_briefs(
    count: int,
    seed: int,
) -> tuple[ConversationGenerationBrief, ...]:
    if not 10 <= count <= 20:
        raise ValueError("A representative pilot requires 10 to 20 conversations.")
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
    required_conditions = ", ".join(brief.required_conditions)
    json_schema = json.dumps(
        EnglishConversationPromptPlan.model_json_schema(),
        separators=(",", ":"),
    )
    return f"""Create one coherent English-only synthetic conversation prompt plan.
Return only one JSON object accepted by the schema at the end of this instruction.

Fixed requirements:
- plan_id is {brief.plan_id!r}; seed is {brief.seed}.
- domain is {brief.domain.value!r}; invent a specific topic unlike generic rural or farming stories.
- target_duration_seconds is 20 to 30 seconds after timing composition.
- include these user conditions: {required_conditions}.
- Use 2 to 6 short user prompts, normally 2 to 45 spoken words each. Prefer ordinary,
  responsive conversation over monologues. A non-floor feedback prompt has at most 8 words.
- Hidden assistant text exists only to make the exchange coherent and estimate duration. Keep each
  assistant turn under 45 words. It will not be synthesized.
- Give the conversation one stable base_user_voice. Vary delivery per user prompt while keeping
  identity stable. The requested anchor delivery is {brief.pace.value} and {brief.affect.value}.
- HOLD text is split into text_before_pause and text_after_pause with a 0.5 to 2.5 second pause.
- Backchannels and reactions reference the assistant turn they occur during. Responses reference
  the assistant turn after which they begin. Interruptions reference the active assistant turn.
- Every interruption is a genuine floor claim and the assistant yields 0.08 to 0.6 seconds later.
- sequence_index values are unique and express semantic ordering, not fabricated audio timestamps.
- All text is natural modern English with printable ASCII punctuation. Do not include stage
  directions, sound effects, copyrighted passages, real public figures, or unsafe content.
- The recording is always clean close-mic speech. Do not request whispering, breathiness,
  background noise, room tone, or environmental audio.

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


def _validate_english_text(value: str) -> str:
    if not value.isascii() or not any(character.isalpha() for character in value):
        raise ValueError("English prompt text must use printable ASCII English text.")
    if any(not character.isprintable() for character in value):
        raise ValueError("English prompt text must use printable ASCII English text.")
    return value


def _validate_short_user_text(value: str, maximum_words: int = 45) -> str:
    _validate_english_text(value)
    if len(value.split()) > maximum_words:
        raise ValueError(f"User prompts require at most {maximum_words} words.")
    return value


def _require_assistant_reference(
    turn_id: str,
    assistant_ids: set[str],
    unit_id: str,
) -> None:
    if turn_id not in assistant_ids:
        raise ValueError(f"User prompt {unit_id} references unknown assistant turn {turn_id}.")


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
