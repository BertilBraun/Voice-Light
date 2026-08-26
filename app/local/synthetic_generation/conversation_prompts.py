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
        reference_texts = tuple(
            " ".join(plan.voice_reference_text.casefold().split()) for plan in self.plans
        )
        if len(reference_texts) != len(set(reference_texts)):
            raise ValueError("Voice reference texts must be unique across a prompt set.")
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
- target_duration_seconds is 60 to 120 seconds after timing composition.
- include these user conditions: {required_conditions}.
- Use 5 to 10 responsive user prompts. Include a brief turn of 2-8 words, a normal turn of 12-28
  words, and an extended turn of 45-90 words targeting 20-30 seconds of rendered user speech.
  Length is derived from text; never output a length_band field. Do not collapse the conversation
  into uniform turn lengths.
- A non_floor_feedback prompt is exactly one item from: yeah, yep, right, okay, sure, mhm, uh-huh,
  or mm-hmm. It cannot contain a clause, evaluation, reaction, or proposition. Collaborative
  completions are excluded from this pilot.
- Hidden assistant text exists only to make the exchange coherent and estimate duration. Keep each
  assistant turn under 45 words. It will not be synthesized.
- Give the conversation one stable base_user_voice. Vary delivery per user prompt while keeping
  identity stable. The requested anchor delivery is {brief.pace.value} and {brief.affect.value}.
- Write voice_reference_text as one exact, neutral English sentence of 8-18 words suitable for a
  clean 3-8 second Qwen VoiceDesign reference render. This exact text will be retained with audio
  and reused to build the conversation's single voice-clone prompt.
- condition must be exactly one of completion, hold, non_floor_feedback,
  response_floor_claim, or interruption_floor_claim. It is never a speech_act.
- speech_act must be exactly one of question, answer, explanation, opinion, anecdote, request,
  or correction. energy must be exactly one of subdued, conversational, or animated.
- Every assistant object includes turn_id, sequence_index, text, speaking_rate_words_per_minute,
  punctuation_pause_seconds, and duration_variation_fraction.
- Every user object includes unit_id, sequence_index, condition, and a delivery object containing
  pace, energy, and affect. Every floor-owning user also includes speech_act. A
  non_floor_feedback deliberately has no speech_act.
- completion includes text. hold includes text_before_pause, text_after_pause, and
  pause_duration_seconds. response_floor_claim includes text, after_assistant_turn_id, and
  response_latency_seconds. interruption_floor_claim includes text, during_assistant_turn_id, and
  assistant_yield_delay_seconds. non_floor_feedback includes text and during_assistant_turn_id.
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
    if any(not character.isprintable() for character in value):
        raise ValueError("English prompt text must use printable text.")
    alphabetic_characters = tuple(character for character in value if character.isalpha())
    if not alphabetic_characters or any(
        not character.isascii() for character in alphabetic_characters
    ):
        raise ValueError("English prompt text must use unaccented Latin letters.")
    return value


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
