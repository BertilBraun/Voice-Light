from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile
from pydantic import ValidationError

from app.local.synthetic_generation.build_conversation_clone_review import render_clone_review
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
    VocalPitch,
    VocalWeight,
)
from app.local.synthetic_generation.conversation_tts import (
    SpeechSynthesisAttemptProvenance,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    VoiceClonePromptProvenance,
    cosyvoice3_reference_prompt,
    load_rendered_user_clips,
    render_conversation_user_audio,
    retain_first_spoken_phrase,
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
from app.local.synthetic_generation.merge_conversation_tts_shards import (
    merge_conversation_tts_shards,
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


def test_cosyvoice3_reference_prompt_separates_control_from_transcript() -> None:
    prompt = cosyvoice3_reference_prompt("This is the exact reference transcript.")

    assert prompt == (
        "You are a helpful assistant.<|endofprompt|>This is the exact reference transcript."
    )

    with pytest.raises(ValueError, match="must not be empty"):
        cosyvoice3_reference_prompt(" ")


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
    assert loaded.references[0].audio_path.suffix == ".flac"
    assert soundfile.info(loaded.references[0].audio_path).format == "FLAC"

    invalid_values = manifest.references[0].model_dump()
    invalid_values["duration_seconds"] = 0.0
    with pytest.raises(ValidationError, match="greater than 0"):
        ConversationVoiceReference.model_validate(invalid_values)
    short_values = manifest.references[0].model_dump()
    short_values["duration_seconds"] = 2.9
    assert ConversationVoiceReference.model_validate(short_values).duration_seconds == 2.9
    longer_values = manifest.references[0].model_dump()
    longer_values["duration_seconds"] = 20.24
    assert ConversationVoiceReference.model_validate(longer_values).duration_seconds == 20.24


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


def test_renderer_measures_natural_tts_hold_without_inserting_silence(tmp_path: Path) -> None:
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
    assert len(requests) == 4
    assert all(
        request.text != "Would you prefer the comedy or the mystery?" for request in requests
    )
    assert all("Speak in English" in request.delivery_instruction for request in requests)
    assert all("conversational affect" in request.delivery_instruction for request in requests)
    assert {request.speed for request in requests} <= {0.96, 1.06, 1.16}
    assert requests[1].text == "mm-hmm"
    assert requests[1].alternative is not None
    assert requests[1].alternative.text == "yeah. I hear you."
    assert requests[1].alternative.retained_prefix_max_seconds == 0.9
    assert len(manifest.rendered_units) == 4
    assert all(unit.reference == reference for unit in manifest.rendered_units)
    hold = next(unit for unit in manifest.rendered_units if unit.prompt.condition == "hold")
    assert len(hold.clauses) == 1
    assert len(hold.clip.continuation_silences) == 1
    measured_pause = hold.clip.continuation_silences[0]
    assert np.isclose(measured_pause.end_seconds - measured_pause.start_seconds, 0.6)
    assert hold.clip.audio_path == Path("audio/pilot_render_user_4.flac")
    loaded = load_rendered_user_clips(tmp_path / "rendered" / "render.json")
    assert soundfile.info(loaded[0].audio_path).format == "FLAC"
    assert all(clip.audio_path.is_absolute() for clip in loaded)
    assert all(
        hashlib.sha256(clip.audio_path.read_bytes()).hexdigest() == clip.audio_sha256
        for clip in loaded
    )

    source_audio_path = tmp_path / "compiled" / "conversations" / "pilot_render" / "source.flac"
    source_audio_path.parent.mkdir(parents=True)
    source_audio_path.write_bytes(loaded[0].audio_path.read_bytes())
    review_path = tmp_path / "review" / "index.html"
    render_clone_review(
        prompts_path=prompt_set_path,
        references_path=reference_manifest_path,
        renders_path=tmp_path / "rendered" / "render.json",
        corpus_directory=tmp_path / "compiled",
        output_path=review_path,
    )
    review = review_path.read_text(encoding="utf-8")
    assert "Qwen VoiceDesign reference" in review
    assert "Complete CosyVoice conversation" in review
    assert "assistant reply" in review
    assert "non_floor_feedback" in review
    assert "../references/audio/pilot_render.flac" in review


def test_attempt_provenance_loads_checkpoint_created_before_text_hashes() -> None:
    attempt = SpeechSynthesisAttemptProvenance.model_validate(
        {
            "attempt_index": 1,
            "seed": 42,
            "generated_duration_seconds": 0.75,
            "generation_seconds": 1.5,
            "outcome": "accepted",
        }
    )

    assert attempt.text_sha256 is None


def test_attempt_provenance_records_empty_model_output() -> None:
    attempt = SpeechSynthesisAttemptProvenance(
        attempt_index=1,
        seed=42,
        text_sha256=hashlib.sha256(b"mm-hmm").hexdigest(),
        generated_duration_seconds=0.0,
        generation_seconds=0.5,
        outcome="empty_model_output",
    )

    assert attempt.generated_duration_seconds == 0.0


def test_retain_first_spoken_phrase_removes_carrier_phrase() -> None:
    sample_rate_hz = 16_000
    first_phrase = np.full(round(0.32 * sample_rate_hz), 0.1, dtype=np.float32)
    pause = np.zeros(round(0.16 * sample_rate_hz), dtype=np.float32)
    carrier_phrase = np.full(round(0.4 * sample_rate_hz), 0.1, dtype=np.float32)

    retained = retain_first_spoken_phrase(
        samples=np.concatenate((first_phrase, pause, carrier_phrase)),
        sample_rate_hz=sample_rate_hz,
        item_id="carrier_phrase",
        maximum_seconds=0.9,
    )

    assert 0.30 <= retained.size / sample_rate_hz <= 0.34


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


def test_renderer_rejects_unknown_or_empty_plan_shard(tmp_path: Path) -> None:
    prompt_set, prompt_set_path = _write_prompt_set(tmp_path)
    reference_manifest, reference_manifest_path = _reference_manifest(
        prompt_set, prompt_set_path, tmp_path
    )

    for plan_ids, expected_message in (
        (frozenset(), "at least one plan"),
        (frozenset({"unknown_plan"}), "unknown plans"),
    ):
        with pytest.raises(ValueError, match=expected_message):
            render_conversation_user_audio(
                prompt_set,
                prompt_set_path,
                reference_manifest,
                reference_manifest_path,
                tmp_path / "rendered",
                RecordingSynthesizer(),
                batch_size=1,
                plan_ids=plan_ids,
            )


def test_merge_conversation_tts_shards_restores_order_and_audio(tmp_path: Path) -> None:
    prompt_set, prompt_set_path = _write_prompt_set(tmp_path)
    reference_manifest, reference_manifest_path = _reference_manifest(
        prompt_set, prompt_set_path, tmp_path
    )
    source_directory = tmp_path / "source"
    complete = render_conversation_user_audio(
        prompt_set,
        prompt_set_path,
        reference_manifest,
        reference_manifest_path,
        source_directory,
        RecordingSynthesizer(),
        batch_size=4,
    )
    shard_paths = []
    for shard_index, shard_units in enumerate(
        (complete.rendered_units[::2], complete.rendered_units[1::2])
    ):
        shard_directory = tmp_path / f"shard-{shard_index}"
        (shard_directory / "audio").mkdir(parents=True)
        for unit in shard_units:
            shutil.copy2(
                source_directory / unit.clip.audio_path,
                shard_directory / unit.clip.audio_path,
            )
        shard_manifest = complete.model_copy(update={"rendered_units": shard_units})
        shard_path = shard_directory / "render.json"
        shard_path.write_text(shard_manifest.model_dump_json(indent=2), encoding="utf-8")
        shard_paths.append(shard_path)

    merged = merge_conversation_tts_shards(
        prompt_set_path,
        reference_manifest_path,
        tuple(shard_paths),
        tmp_path / "merged",
    )

    assert tuple(unit.prompt.unit_id for unit in merged.rendered_units) == (
        "user_1",
        "user_2",
        "user_3",
        "user_4",
    )
    assert all(
        (tmp_path / "merged" / unit.clip.audio_path).exists() for unit in merged.rendered_units
    )
    assert (tmp_path / "merged" / "rendered-units.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 4


def _result(
    request: SpeechSynthesisRequest,
    batch_seed: int,
    clone_prompt: VoiceClonePromptProvenance,
) -> SpeechSynthesisResult:
    return SpeechSynthesisResult(
        clause_id=request.clause_id,
        samples=_speech_samples(include_internal_silence="_user_4_" in request.clause_id),
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
        voice_reference_text=(
            "Tonight I am choosing a cheerful film for our quiet evening indoors while the rain "
            "continues outside."
        ),
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
                text=(
                    "I would rather watch something funny tonight because the entire week has "
                    "already felt much too serious."
                ),
            ),
            NonFloorFeedbackUserPrompt(
                unit_id="user_2",
                sequence_index=2,
                delivery=delivery,
                text=MicroBackchannel.MHM,
                during_assistant_turn_id="assistant_1",
            ),
            CompletionUserPrompt(
                unit_id="user_3",
                sequence_index=3,
                speech_act=SpeechAct.ANSWER,
                delivery=delivery,
                text="The comedy sounds perfect.",
            ),
            HoldUserPrompt(
                unit_id="user_4",
                sequence_index=4,
                speech_act=SpeechAct.EXPLANATION,
                delivery=delivery,
                text=(
                    "The mystery sounds interesting because I usually enjoy following small "
                    "clues and comparing theories before the final reveal arrives, "
                    "but tonight I would rather relax with familiar jokes, warm characters, and "
                    "a story that does not demand too much concentration from either of us."
                    " tonight"
                ),
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


def _speech_samples(include_internal_silence: bool = False) -> np.ndarray:
    sample_rate_hz = 16_000
    leading = np.zeros(round(0.12 * sample_rate_hz), dtype=np.float32)
    speech_seconds = 0.6 if include_internal_silence else 0.5
    time_points = np.arange(round(speech_seconds * sample_rate_hz)) / sample_rate_hz
    speech = (0.2 * np.sin(2.0 * np.pi * 220.0 * time_points)).astype(np.float32)
    if include_internal_silence:
        midpoint = speech.size // 2
        speech = np.concatenate(
            (
                speech[:midpoint],
                np.zeros(round(0.6 * sample_rate_hz), dtype=np.float32),
                speech[midpoint:],
            )
        )
    trailing = np.zeros(round(0.18 * sample_rate_hz), dtype=np.float32)
    return np.concatenate((leading, speech, trailing))


def _reference_samples() -> np.ndarray:
    sample_rate_hz = 16_000
    leading = np.zeros(round(0.12 * sample_rate_hz), dtype=np.float32)
    time_points = np.arange(round(4.0 * sample_rate_hz)) / sample_rate_hz
    speech = (0.2 * np.sin(2.0 * np.pi * 220.0 * time_points)).astype(np.float32)
    trailing = np.zeros(round(0.18 * sample_rate_hz), dtype=np.float32)
    return np.concatenate((leading, speech, trailing))
