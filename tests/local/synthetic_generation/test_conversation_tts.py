from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.conversation_prompts import (
    Affect,
    AssistantTurnPrompt,
    BaseUserVoice,
    CompletionUserPrompt,
    ConversationPromptGeneratorProvenance,
    Energy,
    EnglishAccent,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    HoldUserPrompt,
    MicroBackchannel,
    NonFloorFeedbackUserPrompt,
    PerceivedAge,
    SegmentDelivery,
    SpeakingPace,
    SpeechAct,
    TopicDomain,
    UserTurnLength,
    VocalPitch,
    VocalWeight,
)
from app.local.synthetic_generation.conversation_tts import (
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    VoiceClonePromptProvenance,
    load_rendered_user_clips,
    render_conversation_user_audio,
)
from app.local.synthetic_generation.conversation_voice_references import (
    ConversationVoiceReference,
    ConversationVoiceReferenceManifest,
    ReferenceSynthesisRequest,
    ReferenceSynthesisResult,
    TtsBackendIdentity,
    generate_conversation_voice_references,
    load_voice_reference_manifest,
)


class RecordingSynthesizer:
    def __init__(self) -> None:
        self.requests: list[
            tuple[ConversationVoiceReference, tuple[SpeechSynthesisRequest, ...]]
        ] = []
        self._identity = TtsBackendIdentity(
            backend_id="test-recording-synthesizer",
            model_id="test/model",
            model_revision="test-revision",
            runtime_version="1.0",
            model_license="test-only",
        )

    @property
    def identity(self) -> TtsBackendIdentity:
        return self._identity

    def generate_batch(
        self,
        reference: ConversationVoiceReference,
        requests: tuple[SpeechSynthesisRequest, ...],
    ) -> tuple[SpeechSynthesisResult, ...]:
        self.requests.append((reference, requests))
        clone_prompt = VoiceClonePromptProvenance(
            plan_id=reference.plan_id,
            prompt_sha256="c" * 64,
            reference_audio_sha256=reference.audio_sha256,
            reference_text_sha256=reference.reference_text_sha256,
            backend=self.identity,
        )
        return tuple(
            _result(request, batch_seed=91, clone_prompt=clone_prompt) for request in requests
        )


class RecordingReferenceSynthesizer:
    def __init__(self) -> None:
        self.requests: list[tuple[ReferenceSynthesisRequest, ...]] = []
        self._identity = TtsBackendIdentity(
            backend_id="test-voice-design",
            model_id="test/voice-design",
            model_revision="voice-design-revision",
            runtime_version="1.0",
            model_license="test-only",
        )

    @property
    def identity(self) -> TtsBackendIdentity:
        return self._identity

    def generate_batch(
        self,
        requests: tuple[ReferenceSynthesisRequest, ...],
    ) -> tuple[ReferenceSynthesisResult, ...]:
        self.requests.append(requests)
        return tuple(
            ReferenceSynthesisResult(
                plan_id=request.plan_id,
                samples=_reference_samples(),
                sample_rate_hz=16_000,
                generation_seconds=0.4,
                batch_seed=73,
            )
            for request in requests
        )


def test_reference_stage_uses_exact_plan_text_once(tmp_path: Path) -> None:
    prompt_set, prompt_set_path = _write_prompt_set(tmp_path)
    synthesizer = RecordingReferenceSynthesizer()

    manifest = generate_conversation_voice_references(
        prompt_set,
        prompt_set_path,
        tmp_path / "references",
        synthesizer,
        batch_size=4,
    )

    assert len(synthesizer.requests) == 1
    assert len(synthesizer.requests[0]) == 1
    assert synthesizer.requests[0][0].text == prompt_set.plans[0].voice_reference_text
    assert "English speaker" in synthesizer.requests[0][0].voice_instruction
    assert len(manifest.references) == 1
    assert manifest.references[0].reference_text == prompt_set.plans[0].voice_reference_text
    loaded = load_voice_reference_manifest(tmp_path / "references" / "voice-references.json")
    assert loaded.references[0].audio_path.is_absolute()

    invalid_values = manifest.references[0].model_dump()
    invalid_values["duration_seconds"] = 2.9
    with pytest.raises(ValidationError, match="greater than or equal to 3"):
        ConversationVoiceReference.model_validate(invalid_values)


def test_renderer_rejects_missing_plan_reference(tmp_path: Path) -> None:
    prompt_set, prompt_set_path = _write_prompt_set(tmp_path)
    reference_manifest, reference_manifest_path = _reference_manifest(
        prompt_set, prompt_set_path, tmp_path
    )

    with pytest.raises(ValueError, match=r"missing=\['pilot_render'\]"):
        render_conversation_user_audio(
            prompt_set=prompt_set,
            prompt_set_path=prompt_set_path,
            reference_manifest=reference_manifest.model_copy(update={"references": ()}),
            reference_manifest_path=reference_manifest_path,
            output_directory=tmp_path / "rendered",
            synthesizer=RecordingSynthesizer(),
            batch_size=2,
        )


def test_renderer_batches_user_only_audio_and_composes_exact_hold(tmp_path: Path) -> None:
    prompt_set, prompt_set_path = _write_prompt_set(tmp_path)
    reference_manifest, reference_manifest_path = _reference_manifest(
        prompt_set, prompt_set_path, tmp_path
    )
    synthesizer = RecordingSynthesizer()

    manifest = render_conversation_user_audio(
        prompt_set=prompt_set,
        prompt_set_path=prompt_set_path,
        reference_manifest=reference_manifest,
        reference_manifest_path=reference_manifest_path,
        output_directory=tmp_path / "rendered",
        synthesizer=synthesizer,
        batch_size=4,
    )

    assert len(synthesizer.requests) == 1
    reference, requests = synthesizer.requests[0]
    assert reference.plan_id == "pilot_render"
    assert len(requests) == 5
    assert all(
        request.text != "Would you prefer the comedy or the mystery?" for request in requests
    )
    assert len(manifest.rendered_units) == 4
    assert all(unit.reference == reference for unit in manifest.rendered_units)
    hold = next(unit for unit in manifest.rendered_units if unit.prompt.condition == "hold")
    assert len(hold.clauses) == 2
    assert len(hold.clip.continuation_silences) == 1
    inserted_pause = hold.clip.continuation_silences[0]
    assert np.isclose(inserted_pause.end_seconds - inserted_pause.start_seconds, 0.8)
    assert hold.clip.audio_path == Path("audio/pilot_render_user_4.wav")
    loaded = load_rendered_user_clips(tmp_path / "rendered" / "render.json")
    assert all(clip.audio_path.is_absolute() for clip in loaded)
    assert all(
        hashlib.sha256(clip.audio_path.read_bytes()).hexdigest() == clip.audio_sha256
        for clip in loaded
    )


def test_renderer_resumes_completed_units_without_tts(tmp_path: Path) -> None:
    prompt_set, prompt_set_path = _write_prompt_set(tmp_path)
    reference_manifest, reference_manifest_path = _reference_manifest(
        prompt_set, prompt_set_path, tmp_path
    )
    output_directory = tmp_path / "rendered"
    first = RecordingSynthesizer()
    render_conversation_user_audio(
        prompt_set,
        prompt_set_path,
        reference_manifest,
        reference_manifest_path,
        output_directory,
        first,
        batch_size=1,
    )
    resumed = RecordingSynthesizer()

    manifest = render_conversation_user_audio(
        prompt_set,
        prompt_set_path,
        reference_manifest,
        reference_manifest_path,
        output_directory,
        resumed,
        batch_size=2,
    )

    assert resumed.requests == []
    assert len(manifest.rendered_units) == 4


def _result(
    request: SpeechSynthesisRequest,
    batch_seed: int,
    clone_prompt: VoiceClonePromptProvenance,
) -> SpeechSynthesisResult:
    return SpeechSynthesisResult(
        clause_id=request.clause_id,
        samples=_speech_samples(),
        sample_rate_hz=16_000,
        generation_seconds=0.25,
        batch_seed=batch_seed,
        clone_prompt=clone_prompt,
    )


def _prompt_set() -> EnglishConversationPromptSet:
    delivery = SegmentDelivery(
        pace=SpeakingPace.FAST,
        energy=Energy.ANIMATED,
        affect=Affect.ENGAGING,
    )
    plan = EnglishConversationPromptPlan(
        plan_id="pilot_render",
        seed=81,
        domain=TopicDomain.ENTERTAINMENT,
        topic="Choosing a movie for a rainy evening",
        target_duration_seconds=75.0,
        voice_reference_text="Tonight I am choosing a cheerful film for our quiet evening indoors.",
        base_user_voice=BaseUserVoice(
            perceived_age=PerceivedAge.ADULT,
            accent=EnglishAccent.GENERAL_AMERICAN,
            pitch=VocalPitch.MEDIUM,
            vocal_weight=VocalWeight.MEDIUM,
        ),
        assistant_turns=(
            AssistantTurnPrompt(
                turn_id="assistant_1",
                sequence_index=1,
                text="Would you prefer the comedy or the mystery?",
                speaking_rate_words_per_minute=180,
                punctuation_pause_seconds=0.1,
            ),
        ),
        user_prompts=(
            CompletionUserPrompt(
                unit_id="user_1",
                sequence_index=0,
                speech_act=SpeechAct.OPINION,
                delivery=delivery,
                length_band=UserTurnLength.NORMAL,
                text=(
                    "I would rather watch something funny tonight because the entire week has "
                    "already felt much too serious."
                ),
            ),
            NonFloorFeedbackUserPrompt(
                unit_id="user_2",
                sequence_index=2,
                speech_act=SpeechAct.ANSWER,
                delivery=delivery,
                text=MicroBackchannel.RIGHT,
                during_assistant_turn_id="assistant_1",
            ),
            CompletionUserPrompt(
                unit_id="user_3",
                sequence_index=3,
                speech_act=SpeechAct.ANSWER,
                delivery=delivery,
                length_band=UserTurnLength.BRIEF,
                text="The comedy sounds perfect.",
            ),
            HoldUserPrompt(
                unit_id="user_4",
                sequence_index=4,
                speech_act=SpeechAct.EXPLANATION,
                delivery=delivery,
                length_band=UserTurnLength.EXTENDED,
                text_before_pause=(
                    "The mystery sounds interesting because I usually enjoy following small "
                    "clues and comparing theories before the final reveal arrives"
                ),
                text_after_pause=(
                    "but tonight I would rather relax with familiar jokes, warm characters, and "
                    "a story that does not demand too much concentration from either of us."
                    " tonight"
                ),
                pause_duration_seconds=0.8,
            ),
        ),
    )
    return EnglishConversationPromptSet(
        set_id="render_pilot",
        provenance=ConversationPromptGeneratorProvenance(
            model_id="test/prompt-model",
            model_revision="a" * 40,
            runtime_version="1.0",
            seed=81,
            requested_plan_count=10,
        ),
        plans=(plan,),
    )


def _write_prompt_set(tmp_path: Path) -> tuple[EnglishConversationPromptSet, Path]:
    prompt_set = _prompt_set()
    prompt_set_path = tmp_path / "prompts.json"
    prompt_set_path.write_text(prompt_set.model_dump_json(indent=2), encoding="utf-8")
    return prompt_set, prompt_set_path


def _reference_manifest(
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    tmp_path: Path,
) -> tuple[ConversationVoiceReferenceManifest, Path]:
    reference_manifest_path = tmp_path / "references" / "voice-references.json"
    generate_conversation_voice_references(
        prompt_set,
        prompt_set_path,
        reference_manifest_path.parent,
        RecordingReferenceSynthesizer(),
        batch_size=1,
    )
    return load_voice_reference_manifest(reference_manifest_path), reference_manifest_path


def _speech_samples() -> np.ndarray:
    sample_rate_hz = 16_000
    leading = np.zeros(round(0.12 * sample_rate_hz), dtype=np.float32)
    time_points = np.arange(round(0.5 * sample_rate_hz)) / sample_rate_hz
    speech = (0.2 * np.sin(2.0 * np.pi * 220.0 * time_points)).astype(np.float32)
    trailing = np.zeros(round(0.18 * sample_rate_hz), dtype=np.float32)
    return np.concatenate((leading, speech, trailing))


def _reference_samples() -> np.ndarray:
    sample_rate_hz = 16_000
    leading = np.zeros(round(0.12 * sample_rate_hz), dtype=np.float32)
    time_points = np.arange(round(4.0 * sample_rate_hz)) / sample_rate_hz
    speech = (0.2 * np.sin(2.0 * np.pi * 220.0 * time_points)).astype(np.float32)
    trailing = np.zeros(round(0.18 * sample_rate_hz), dtype=np.float32)
    return np.concatenate((leading, speech, trailing))
