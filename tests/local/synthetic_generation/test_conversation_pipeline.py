from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from app.local.synthetic_generation.completion_dataset import DEFAULT_SILENCE_DETECTION
from app.local.synthetic_generation.conversation_compiler import (
    ConversationCompilerConfig,
    RenderedUserClip,
)
from app.local.synthetic_generation.conversation_pipeline import build_conversation_corpus
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
    InterruptionFloorClaimUserPrompt,
    NonFloorFeedbackKind,
    NonFloorFeedbackUserPrompt,
    PerceivedAge,
    ResponseFloorClaimUserPrompt,
    SegmentDelivery,
    SpeakingPace,
    SpeechAct,
    TopicDomain,
    VocalPitch,
    VocalWeight,
)
from app.local.synthetic_generation.conversation_tts import (
    ConversationTtsManifest,
    RenderedClauseProvenance,
    RenderedConversationUserUnit,
    TtsBackendIdentity,
)
from app.local.training_corpus.export import MaterializedTrainingSample


def test_build_conversation_corpus_exports_user_only_multievent_training_rows(
    tmp_path: Path,
) -> None:
    prompt_set = _prompt_set()
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(prompt_set.model_dump_json(indent=2), encoding="utf-8")
    render_directory = tmp_path / "render"
    backend = TtsBackendIdentity(
        backend_id="fake",
        model_id="fake/tts",
        model_revision="test",
        runtime_version="1",
        model_license="test-only",
    )
    rendered_units = tuple(
        _rendered_unit(render_directory, prompt_set.plans[0], prompt, backend)
        for prompt in prompt_set.plans[0].user_prompts
    )
    tts_manifest = ConversationTtsManifest(
        prompt_set_id=prompt_set.set_id,
        prompt_set_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        backend=backend,
        detection=DEFAULT_SILENCE_DETECTION,
        rendered_units=rendered_units,
    )
    tts_path = render_directory / "render.json"
    tts_path.parent.mkdir(parents=True, exist_ok=True)
    tts_path.write_text(tts_manifest.model_dump_json(indent=2), encoding="utf-8")

    manifest = build_conversation_corpus(
        prompt_set_path=prompt_path,
        tts_manifest_path=tts_path,
        output_directory=tmp_path / "corpus",
        split_seed="test-split",
        compiler_config=ConversationCompilerConfig(
            crop_variant_count=3,
            event_light_fraction=0.0,
            assistant_duration_variation=0.0,
        ),
    )

    assert len(manifest.conversations) == 1
    shard = next((tmp_path / "corpus" / "training").rglob("*.parquet"))
    rows = pq.read_table(shard).to_pylist()
    samples = tuple(MaterializedTrainingSample.model_validate(row) for row in rows)
    assert len(samples) == 3
    assert all(sample.assistant_audio_path is None for sample in samples)
    assert all(sample.p_user_floor_now is not None for sample in samples)
    assert all(sample.speculative_eot_500 is not None for sample in samples)
    assert all(sample.p_user_yield == sample.yield_oriented_primary_targets() for sample in samples)
    assert any(value == 1.0 for sample in samples for value in sample.non_floor_feedback)
    assert any(value == 1.0 for sample in samples for value in sample.floor_take)
    compiled_path = tmp_path / "corpus" / manifest.conversations[0].manifest_path
    compiled_payload = compiled_path.read_text(encoding="utf-8")
    assert "assistant_audio" not in compiled_payload


def _prompt_set() -> EnglishConversationPromptSet:
    delivery = SegmentDelivery(
        pace=SpeakingPace.FAST,
        energy=Energy.ANIMATED,
        affect=Affect.ENGAGING,
    )
    plan = EnglishConversationPromptPlan(
        plan_id="pipeline_pilot",
        seed=71,
        domain=TopicDomain.TECHNOLOGY,
        topic="Choosing notification settings for a shared calendar",
        target_duration_seconds=24.0,
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
                text=(
                    "I can compare the reminder choices and explain which one is less distracting."
                ),
                speaking_rate_words_per_minute=180,
                punctuation_pause_seconds=0.1,
            ),
            AssistantTurnPrompt(
                turn_id="assistant_2",
                sequence_index=4,
                text="The shared calendar also supports a separate alert for urgent changes.",
                speaking_rate_words_per_minute=175,
                punctuation_pause_seconds=0.1,
            ),
        ),
        user_prompts=(
            CompletionUserPrompt(
                unit_id="user_1",
                sequence_index=0,
                speech_act=SpeechAct.REQUEST,
                delivery=delivery,
                text="Help me choose a calmer reminder setting.",
            ),
            NonFloorFeedbackUserPrompt(
                unit_id="user_2",
                sequence_index=2,
                speech_act=SpeechAct.ANSWER,
                delivery=delivery,
                text="Right.",
                feedback_kind=NonFloorFeedbackKind.BACKCHANNEL,
                during_assistant_turn_id="assistant_1",
            ),
            ResponseFloorClaimUserPrompt(
                unit_id="user_3",
                sequence_index=3,
                speech_act=SpeechAct.OPINION,
                delivery=delivery,
                text="The quieter option sounds better.",
                after_assistant_turn_id="assistant_1",
                response_latency_seconds=0.3,
            ),
            InterruptionFloorClaimUserPrompt(
                unit_id="user_4",
                sequence_index=5,
                speech_act=SpeechAct.CORRECTION,
                delivery=delivery,
                text="Wait, urgent changes should still make a sound.",
                during_assistant_turn_id="assistant_2",
                assistant_yield_delay_seconds=0.2,
            ),
        ),
    )
    return EnglishConversationPromptSet(
        set_id="pipeline_test",
        provenance=ConversationPromptGeneratorProvenance(
            model_id="test/planner",
            model_revision="a" * 40,
            runtime_version="1",
            seed=71,
            requested_plan_count=10,
        ),
        plans=(plan,),
    )


def _rendered_unit(
    directory: Path,
    plan: EnglishConversationPromptPlan,
    prompt: (
        CompletionUserPrompt
        | HoldUserPrompt
        | NonFloorFeedbackUserPrompt
        | ResponseFloorClaimUserPrompt
        | InterruptionFloorClaimUserPrompt
    ),
    backend: TtsBackendIdentity,
) -> RenderedConversationUserUnit:
    clip_id = f"{plan.plan_id}_{prompt.unit_id}"
    relative_path = Path("audio") / f"{clip_id}.wav"
    audio_path = directory / relative_path
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate_hz = 16_000
    match prompt:
        case NonFloorFeedbackUserPrompt():
            duration_seconds = 0.45
        case _:
            duration_seconds = 1.0
    times = np.arange(round(duration_seconds * sample_rate_hz)) / sample_rate_hz
    samples = (np.sin(2.0 * np.pi * 220.0 * times) * 0.1 * 32767.0).astype("<i2")
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate_hz)
        audio_file.writeframes(samples.tobytes())
    audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    clip = RenderedUserClip(
        clip_id=clip_id,
        audio_path=relative_path,
        audio_sha256=audio_sha256,
        duration_seconds=duration_seconds,
        active_start_seconds=0.0,
        active_end_seconds=duration_seconds,
        continuation_silences=(),
    )
    clause = RenderedClauseProvenance(
        clause_id=f"{clip_id}_clause_1",
        text_sha256="b" * 64,
        request_seed=1,
        batch_seed=1,
        sample_rate_hz=sample_rate_hz,
        original_duration_seconds=duration_seconds,
        trimmed_duration_seconds=duration_seconds,
        trimmed_leading_seconds=0.0,
        trimmed_trailing_seconds=0.0,
        generation_seconds=0.1,
        real_time_factor=0.1,
    )
    return RenderedConversationUserUnit(
        plan_id=plan.plan_id,
        prompt=prompt,
        base_user_voice=plan.base_user_voice,
        backend=backend,
        voice_instruction="Clean English test voice.",
        prompt_sha256="c" * 64,
        clip=clip,
        clauses=(clause,),
    )
