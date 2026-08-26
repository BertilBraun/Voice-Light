from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

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
    PerceivedAge,
    SegmentDelivery,
    SpeakingPace,
    SpeechAct,
    TopicDomain,
    VocalPitch,
    VocalWeight,
)
from app.local.synthetic_generation.conversation_tts import (
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    TtsBackendIdentity,
    load_rendered_user_clips,
    render_conversation_user_audio,
)


class RecordingSynthesizer:
    def __init__(self) -> None:
        self.requests: list[tuple[SpeechSynthesisRequest, ...]] = []
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
        requests: tuple[SpeechSynthesisRequest, ...],
    ) -> tuple[SpeechSynthesisResult, ...]:
        self.requests.append(requests)
        return tuple(_result(request, batch_seed=91) for request in requests)


def test_renderer_batches_user_only_audio_and_composes_exact_hold(tmp_path: Path) -> None:
    prompt_set = _prompt_set()
    prompt_set_path = tmp_path / "prompts.json"
    prompt_set_path.write_text(prompt_set.model_dump_json(indent=2), encoding="utf-8")
    synthesizer = RecordingSynthesizer()

    manifest = render_conversation_user_audio(
        prompt_set=prompt_set,
        prompt_set_path=prompt_set_path,
        output_directory=tmp_path / "rendered",
        synthesizer=synthesizer,
        batch_size=2,
    )

    assert len(synthesizer.requests) == 1
    assert len(synthesizer.requests[0]) == 3
    assert all(
        request.text != "Would you prefer the comedy or the mystery?"
        for request in synthesizer.requests[0]
    )
    assert all(
        request.voice_instruction == synthesizer.requests[0][0].voice_instruction
        for request in synthesizer.requests[0]
    )
    assert all(
        "English speaker" in request.voice_instruction for request in synthesizer.requests[0]
    )
    assert len(manifest.rendered_units) == 2
    hold = next(unit for unit in manifest.rendered_units if unit.prompt.condition == "hold")
    assert len(hold.clauses) == 2
    assert len(hold.clip.continuation_silences) == 1
    inserted_pause = hold.clip.continuation_silences[0]
    assert np.isclose(inserted_pause.end_seconds - inserted_pause.start_seconds, 0.8)
    assert hold.clip.audio_path == Path("audio/pilot_render_user_2.wav")
    loaded = load_rendered_user_clips(tmp_path / "rendered" / "render.json")
    assert all(clip.audio_path.is_absolute() for clip in loaded)
    assert all(
        hashlib.sha256(clip.audio_path.read_bytes()).hexdigest() == clip.audio_sha256
        for clip in loaded
    )


def test_renderer_resumes_completed_units_without_tts(tmp_path: Path) -> None:
    prompt_set = _prompt_set()
    prompt_set_path = tmp_path / "prompts.json"
    prompt_set_path.write_text(prompt_set.model_dump_json(), encoding="utf-8")
    output_directory = tmp_path / "rendered"
    first = RecordingSynthesizer()
    render_conversation_user_audio(
        prompt_set,
        prompt_set_path,
        output_directory,
        first,
        batch_size=1,
    )
    resumed = RecordingSynthesizer()

    manifest = render_conversation_user_audio(
        prompt_set,
        prompt_set_path,
        output_directory,
        resumed,
        batch_size=2,
    )

    assert resumed.requests == []
    assert len(manifest.rendered_units) == 2


def _result(request: SpeechSynthesisRequest, batch_seed: int) -> SpeechSynthesisResult:
    sample_rate_hz = 16_000
    leading = np.zeros(round(0.12 * sample_rate_hz), dtype=np.float32)
    time_points = np.arange(round(0.5 * sample_rate_hz)) / sample_rate_hz
    speech = (0.2 * np.sin(2.0 * np.pi * 220.0 * time_points)).astype(np.float32)
    trailing = np.zeros(round(0.18 * sample_rate_hz), dtype=np.float32)
    return SpeechSynthesisResult(
        clause_id=request.clause_id,
        samples=np.concatenate((leading, speech, trailing)),
        sample_rate_hz=sample_rate_hz,
        generation_seconds=0.25,
        batch_seed=batch_seed,
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
        target_duration_seconds=22.0,
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
                text="I would rather watch something funny tonight.",
            ),
            HoldUserPrompt(
                unit_id="user_2",
                sequence_index=2,
                speech_act=SpeechAct.EXPLANATION,
                delivery=delivery,
                text_before_pause="The mystery sounds interesting",
                text_after_pause="but I am not in the mood for anything grim.",
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
