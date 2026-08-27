from __future__ import annotations

import hashlib
import html
import wave
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from app.shared.base_model import FrozenBaseModel


class ListeningModel(FrozenBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class ListeningProvider(str, Enum):
    QWEN_CUSTOM = "qwen_custom"
    COSYVOICE3 = "cosyvoice3"
    INDEXTTS25 = "indextts25"


class ListeningPurpose(str, Enum):
    BACKCHANNEL = "backchannel"
    SHORT_TURN = "short_turn"
    NORMAL_TURN = "normal_turn"
    LONG_TURN = "long_turn"


class VoiceKey(str, Enum):
    QWEN_RYAN = "qwen_ryan"
    QWEN_AIDEN = "qwen_aiden"
    QWEN_VIVIAN = "qwen_vivian"
    QWEN_SOHEE = "qwen_sohee"
    REFERENCE_ONE = "reference_one"
    REFERENCE_TWO = "reference_two"


class ListeningUtterance(ListeningModel):
    utterance_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    purpose: ListeningPurpose
    voice: VoiceKey
    text: str = Field(min_length=1)
    delivery_instruction: str = Field(min_length=1)


class ConversationTurn(ListeningModel):
    utterance_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    assistant_gap_after_seconds: float = Field(ge=0.0, le=8.0)


class ListeningConversation(ListeningModel):
    conversation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    turns: tuple[ConversationTurn, ...] = Field(min_length=2)


class ProviderListeningPlan(ListeningModel):
    schema_version: Literal["voice-light-provider-listening-plan-v1"] = (
        "voice-light-provider-listening-plan-v1"
    )
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    provider: ListeningProvider
    utterances: tuple[ListeningUtterance, ...] = Field(min_length=1)
    conversations: tuple[ListeningConversation, ...] = ()

    @model_validator(mode="after")
    def validate_bindings(self) -> ProviderListeningPlan:
        utterances_by_id = {item.utterance_id: item for item in self.utterances}
        if len(utterances_by_id) != len(self.utterances):
            raise ValueError("Listening utterance IDs must be unique.")
        conversation_ids = {conversation.conversation_id for conversation in self.conversations}
        if len(conversation_ids) != len(self.conversations):
            raise ValueError("Listening conversation IDs must be unique.")
        for conversation in self.conversations:
            for turn in conversation.turns:
                if turn.utterance_id not in utterances_by_id:
                    raise ValueError(
                        f"Conversation {conversation.conversation_id} references "
                        "an unknown utterance."
                    )
        return self


class RenderedListeningUtterance(ListeningModel):
    utterance: ListeningUtterance
    speaker_label: str = Field(min_length=1)
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0)
    generation_seconds: float = Field(ge=0.0)
    real_time_factor: float = Field(ge=0.0)


class RenderedConversationTurn(ListeningModel):
    utterance_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    speech_start_seconds: float = Field(ge=0.0)
    speech_end_seconds: float = Field(gt=0.0)
    assistant_start_seconds: float = Field(ge=0.0)
    assistant_end_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_timeline(self) -> RenderedConversationTurn:
        if self.speech_end_seconds <= self.speech_start_seconds:
            raise ValueError("Conversation speech must have positive duration.")
        if self.assistant_start_seconds != self.speech_end_seconds:
            raise ValueError("Assistant activity must begin when user speech ends.")
        if self.assistant_end_seconds < self.assistant_start_seconds:
            raise ValueError("Assistant activity cannot have negative duration.")
        return self


class RenderedListeningConversation(ListeningModel):
    conversation: ListeningConversation
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0)
    timeline: tuple[RenderedConversationTurn, ...] = Field(min_length=2)


class ProviderListeningManifest(ListeningModel):
    schema_version: Literal["voice-light-provider-listening-results-v1"] = (
        "voice-light-provider-listening-results-v1"
    )
    plan: ProviderListeningPlan
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    rendered_utterances: tuple[RenderedListeningUtterance, ...]
    rendered_conversations: tuple[RenderedListeningConversation, ...]


def load_provider_listening_manifest(path: Path) -> ProviderListeningManifest:
    manifest = ProviderListeningManifest.model_validate_json(path.read_text(encoding="utf-8"))
    return manifest.model_copy(
        update={
            "rendered_utterances": tuple(
                rendered.model_copy(update={"audio_path": path.parent / rendered.audio_path})
                for rendered in manifest.rendered_utterances
            ),
            "rendered_conversations": tuple(
                rendered.model_copy(update={"audio_path": path.parent / rendered.audio_path})
                for rendered in manifest.rendered_conversations
            ),
        }
    )


def write_provider_listening_manifest(
    manifest: ProviderListeningManifest,
    output_directory: Path,
) -> Path:
    relative_manifest = manifest.model_copy(
        update={
            "rendered_utterances": tuple(
                rendered.model_copy(
                    update={"audio_path": rendered.audio_path.relative_to(output_directory)}
                )
                for rendered in manifest.rendered_utterances
            ),
            "rendered_conversations": tuple(
                rendered.model_copy(
                    update={"audio_path": rendered.audio_path.relative_to(output_directory)}
                )
                for rendered in manifest.rendered_conversations
            ),
        }
    )
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(relative_manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest_path


def qwen_custom_listening_plan() -> ProviderListeningPlan:
    return ProviderListeningPlan(
        plan_id="qwen_custom_english_voices_v1",
        provider=ListeningProvider.QWEN_CUSTOM,
        utterances=(
            _utterance(
                "ryan_engaged",
                ListeningPurpose.NORMAL_TURN,
                VoiceKey.QWEN_RYAN,
                "I found the problem. The calendar was using the old time zone, "
                "so every meeting shifted by an hour.",
                "Sound upbeat, engaged, and conversational, with brisk but clear pacing.",
            ),
            _utterance(
                "ryan_mysterious",
                ListeningPurpose.NORMAL_TURN,
                VoiceKey.QWEN_RYAN,
                "The strange part is that the light only appears after everyone "
                "has left the building.",
                "Speak quietly and mysteriously, with measured pacing and restrained tension.",
            ),
            _utterance(
                "aiden_fast",
                ListeningPurpose.NORMAL_TURN,
                VoiceKey.QWEN_AIDEN,
                "We can catch the earlier train if we leave now, grab coffee at the station, "
                "and buy the tickets on the platform.",
                "Speak quickly, brightly, and confidently while remaining natural "
                "and intelligible.",
            ),
            _utterance(
                "aiden_calm",
                ListeningPurpose.NORMAL_TURN,
                VoiceKey.QWEN_AIDEN,
                "I would start with the smaller room, because we can finish it today "
                "and learn what the larger repair will involve.",
                "Use a calm, thoughtful, practical conversational delivery at a moderate pace.",
            ),
            _utterance(
                "vivian_engaged",
                ListeningPurpose.SHORT_TURN,
                VoiceKey.QWEN_VIVIAN,
                "That actually worked much better than I expected.",
                "Sound pleasantly surprised, engaged, and natural in English.",
            ),
            _utterance(
                "vivian_neutral",
                ListeningPurpose.SHORT_TURN,
                VoiceKey.QWEN_VIVIAN,
                "I left the spare key beside the blue planter.",
                "Use a clear, relaxed, neutral English conversational tone.",
            ),
            _utterance(
                "sohee_warm",
                ListeningPurpose.SHORT_TURN,
                VoiceKey.QWEN_SOHEE,
                "Yes, I would love to come with you this weekend.",
                "Sound warm, friendly, and genuinely enthusiastic in English.",
            ),
            _utterance(
                "sohee_concerned",
                ListeningPurpose.SHORT_TURN,
                VoiceKey.QWEN_SOHEE,
                "Wait, are you sure that door was locked when we arrived?",
                "Sound quietly concerned and alert, without becoming theatrical.",
            ),
        ),
    )


def cloning_provider_listening_plan(provider: ListeningProvider) -> ProviderListeningPlan:
    if provider not in {ListeningProvider.COSYVOICE3, ListeningProvider.INDEXTTS25}:
        raise ValueError("The cloning listening plan requires a cloning provider.")
    utterances = (
        _utterance(
            "backchannel_yeah",
            ListeningPurpose.BACKCHANNEL,
            VoiceKey.REFERENCE_ONE,
            "Yeah.",
            "A brief, casual listener acknowledgment.",
        ),
        _utterance(
            "backchannel_mhm",
            ListeningPurpose.BACKCHANNEL,
            VoiceKey.REFERENCE_TWO,
            "Mm-hmm.",
            "A very short, natural listener acknowledgment.",
        ),
        _utterance(
            "backchannel_right",
            ListeningPurpose.BACKCHANNEL,
            VoiceKey.REFERENCE_ONE,
            "Right.",
            "A quick, attentive listener acknowledgment.",
        ),
        _utterance(
            "backchannel_okay",
            ListeningPurpose.BACKCHANNEL,
            VoiceKey.REFERENCE_TWO,
            "Okay.",
            "A short, neutral listener acknowledgment.",
        ),
        _utterance(
            "backchannel_sure",
            ListeningPurpose.BACKCHANNEL,
            VoiceKey.REFERENCE_ONE,
            "Sure.",
            "A brief, agreeable listener acknowledgment.",
        ),
        _utterance(
            "turn_short_key",
            ListeningPurpose.SHORT_TURN,
            VoiceKey.REFERENCE_ONE,
            "I left the spare key beside the blue planter.",
            "Clear, relaxed, and conversational.",
        ),
        _utterance(
            "turn_medium_update",
            ListeningPurpose.NORMAL_TURN,
            VoiceKey.REFERENCE_TWO,
            "The update fixed the connection problem, but I still need to test whether "
            "notifications arrive when the screen is locked.",
            "Thoughtful and matter-of-fact at a natural pace.",
        ),
        _utterance(
            "turn_long_trip",
            ListeningPurpose.LONG_TURN,
            VoiceKey.REFERENCE_ONE,
            "We took the coastal road because the highway was backed up, and it added about "
            "twenty minutes, but the view was worth it. We stopped near the lighthouse, "
            "walked down to the water, and still reached the hotel before dinner.",
            "Animated and engaged, with naturally varied pacing.",
        ),
        _utterance(
            "conversation_one_open",
            ListeningPurpose.NORMAL_TURN,
            VoiceKey.REFERENCE_ONE,
            "I tried the new bakery this morning, and their sourdough was genuinely excellent.",
            "Friendly and conversational.",
        ),
        _utterance(
            "conversation_one_backchannel",
            ListeningPurpose.BACKCHANNEL,
            VoiceKey.REFERENCE_ONE,
            "Yeah.",
            "A very short listener backchannel.",
        ),
        _utterance(
            "conversation_one_close",
            ListeningPurpose.NORMAL_TURN,
            VoiceKey.REFERENCE_ONE,
            "Exactly, and they start selling out before lunch, so we should probably "
            "go early on Saturday.",
            "Engaged and lightly enthusiastic.",
        ),
        _utterance(
            "conversation_two_open",
            ListeningPurpose.SHORT_TURN,
            VoiceKey.REFERENCE_TWO,
            "Did the mechanic figure out what caused that rattling sound?",
            "Curious and mildly concerned.",
        ),
        _utterance(
            "conversation_two_backchannel",
            ListeningPurpose.BACKCHANNEL,
            VoiceKey.REFERENCE_TWO,
            "Mm-hmm.",
            "A very short attentive backchannel.",
        ),
        _utterance(
            "conversation_two_close",
            ListeningPurpose.NORMAL_TURN,
            VoiceKey.REFERENCE_TWO,
            "That makes sense. If the replacement part arrives tomorrow, I can collect the "
            "car after work instead of rearranging the whole afternoon.",
            "Relieved, practical, and conversational.",
        ),
    )
    return ProviderListeningPlan(
        plan_id=f"{provider.value}_english_clone_v1",
        provider=provider,
        utterances=utterances,
        conversations=(
            ListeningConversation(
                conversation_id="bakery_follow_up",
                title="Bakery recommendation",
                turns=(
                    ConversationTurn(
                        utterance_id="conversation_one_open", assistant_gap_after_seconds=2.8
                    ),
                    ConversationTurn(
                        utterance_id="conversation_one_backchannel", assistant_gap_after_seconds=2.2
                    ),
                    ConversationTurn(
                        utterance_id="conversation_one_close", assistant_gap_after_seconds=0.0
                    ),
                ),
            ),
            ListeningConversation(
                conversation_id="mechanic_follow_up",
                title="Car repair follow-up",
                turns=(
                    ConversationTurn(
                        utterance_id="conversation_two_open", assistant_gap_after_seconds=3.0
                    ),
                    ConversationTurn(
                        utterance_id="conversation_two_backchannel", assistant_gap_after_seconds=2.5
                    ),
                    ConversationTurn(
                        utterance_id="conversation_two_close", assistant_gap_after_seconds=0.0
                    ),
                ),
            ),
        ),
    )


def compose_listening_conversations(
    plan: ProviderListeningPlan,
    rendered_utterances: tuple[RenderedListeningUtterance, ...],
    output_directory: Path,
) -> tuple[RenderedListeningConversation, ...]:
    rendered_by_id = {rendered.utterance.utterance_id: rendered for rendered in rendered_utterances}
    if len(rendered_by_id) != len(rendered_utterances):
        raise ValueError("Rendered listening utterance IDs must be unique.")
    output_directory.mkdir(parents=True, exist_ok=True)
    conversations = []
    for conversation in plan.conversations:
        sample_rate_hz: int | None = None
        chunks: list[np.ndarray] = []
        timeline = []
        cursor_seconds = 0.0
        for turn in conversation.turns:
            rendered = rendered_by_id[turn.utterance_id]
            samples, turn_sample_rate_hz = read_pcm16_wave(rendered.audio_path)
            if sample_rate_hz is None:
                sample_rate_hz = turn_sample_rate_hz
            if turn_sample_rate_hz != sample_rate_hz:
                raise ValueError("Conversation utterances must use the same sample rate.")
            speech_start_seconds = cursor_seconds
            speech_end_seconds = speech_start_seconds + samples.size / sample_rate_hz
            assistant_end_seconds = speech_end_seconds + turn.assistant_gap_after_seconds
            timeline.append(
                RenderedConversationTurn(
                    utterance_id=turn.utterance_id,
                    speech_start_seconds=speech_start_seconds,
                    speech_end_seconds=speech_end_seconds,
                    assistant_start_seconds=speech_end_seconds,
                    assistant_end_seconds=assistant_end_seconds,
                )
            )
            chunks.append(samples)
            gap_sample_count = round(turn.assistant_gap_after_seconds * sample_rate_hz)
            if gap_sample_count:
                chunks.append(np.zeros(gap_sample_count, dtype=np.float32))
            cursor_seconds = assistant_end_seconds
        assert sample_rate_hz is not None
        audio_path = output_directory / f"{conversation.conversation_id}.wav"
        write_pcm16_wave(audio_path, np.concatenate(chunks), sample_rate_hz)
        conversations.append(
            RenderedListeningConversation(
                conversation=conversation,
                audio_path=audio_path,
                audio_sha256=file_sha256(audio_path),
                sample_rate_hz=sample_rate_hz,
                duration_seconds=cursor_seconds,
                timeline=tuple(timeline),
            )
        )
    return tuple(conversations)


def write_pcm16_wave(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as output_wave:
        output_wave.setnchannels(1)
        output_wave.setsampwidth(2)
        output_wave.setframerate(sample_rate_hz)
        output_wave.writeframes((pcm * 32767.0).astype("<i2").tobytes())


def read_pcm16_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as input_wave:
        if input_wave.getnchannels() != 1 or input_wave.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16 audio: {path}")
        sample_rate_hz = input_wave.getframerate()
        samples = np.frombuffer(input_wave.readframes(input_wave.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0, sample_rate_hz


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_listening_review(
    manifests: tuple[ProviderListeningManifest, ...],
    output_path: Path,
) -> None:
    sections = "".join(_provider_section(manifest, output_path.parent) for manifest in manifests)
    document = "".join(
        (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>Voice Light TTS provider listening pilot</title><style>",
            "body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;",
            "background:#111827;color:#e5e7eb}h1,h2,h3{color:#f9fafb}",
            "section,article{background:#1f2937;border:1px solid #374151;",
            "border-radius:12px;padding:16px;margin:14px 0}audio{width:100%}",
            ".meta{color:#9ca3af}.tag{display:inline-block;background:#374151;",
            "border-radius:999px;padding:3px 9px;margin-right:6px}</style></head><body>",
            "<h1>Voice Light TTS provider listening pilot</h1>",
            "<p>Raw clips isolate synthesis quality. Conversation WAVs contain silent ",
            "intervals representing assistant turns; these are not user holds.</p>",
            sections,
            "</body></html>",
        )
    )
    output_path.write_text(document, encoding="utf-8")


def _provider_section(manifest: ProviderListeningManifest, review_directory: Path) -> str:
    utterance_cards = "".join(
        _utterance_card(rendered, review_directory) for rendered in manifest.rendered_utterances
    )
    conversation_cards = "".join(
        _conversation_card(rendered, review_directory)
        for rendered in manifest.rendered_conversations
    )
    conversation_content = conversation_cards or "<p>None for this preset-voice audition.</p>"
    return "".join(
        (
            f"<section><h2>{html.escape(manifest.plan.provider.value)}</h2>",
            f"<p class='meta'>{html.escape(manifest.model_id)} @ ",
            f"{html.escape(manifest.model_revision)}</p><h3>Raw clips</h3>",
            utterance_cards,
            "<h3>Composed conversations</h3>",
            conversation_content,
            "</section>",
        )
    )


def _utterance_card(rendered: RenderedListeningUtterance, review_directory: Path) -> str:
    relative_path = rendered.audio_path.relative_to(review_directory).as_posix()
    utterance = rendered.utterance
    return "".join(
        (
            f"<article><span class='tag'>{html.escape(utterance.purpose.value)}</span>",
            f"<span class='tag'>{html.escape(rendered.speaker_label)}</span>",
            f"<p>{html.escape(utterance.text)}</p><p class='meta'>",
            f"{html.escape(utterance.delivery_instruction)} · ",
            f"{rendered.duration_seconds:.2f}s · RTF {rendered.real_time_factor:.2f}</p>",
            f"<audio controls preload='none' src='{html.escape(relative_path)}'>",
            "</audio></article>",
        )
    )


def _conversation_card(rendered: RenderedListeningConversation, review_directory: Path) -> str:
    relative_path = rendered.audio_path.relative_to(review_directory).as_posix()
    timeline = " · ".join(
        f"{turn.utterance_id} {turn.speech_start_seconds:.1f}–"
        f"{turn.speech_end_seconds:.1f}s; assistant until "
        f"{turn.assistant_end_seconds:.1f}s"
        for turn in rendered.timeline
    )
    return "".join(
        (
            f"<article><h3>{html.escape(rendered.conversation.title)}</h3>",
            f"<p class='meta'>{html.escape(timeline)}</p>",
            f"<audio controls preload='none' src='{html.escape(relative_path)}'>",
            "</audio></article>",
        )
    )


def _utterance(
    utterance_id: str,
    purpose: ListeningPurpose,
    voice: VoiceKey,
    text: str,
    delivery_instruction: str,
) -> ListeningUtterance:
    return ListeningUtterance(
        utterance_id=utterance_id,
        purpose=purpose,
        voice=voice,
        text=text,
        delivery_instruction=delivery_instruction,
    )
