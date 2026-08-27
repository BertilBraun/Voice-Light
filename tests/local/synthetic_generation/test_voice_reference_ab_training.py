from __future__ import annotations

from pathlib import Path

import numpy as np

from app.local.synthetic_generation.conversation_compiler import (
    AfterUserAssistantTiming,
    ConversationCompilerConfig,
    materialize_crop_audio,
)
from app.local.synthetic_generation.voice_reference_ab_pilot import (
    audition_utterances,
    materialize_trimmed_audition_utterance,
)
from app.local.synthetic_generation.voice_reference_ab_training import (
    compile_audition_candidate_training_samples,
)


def test_compiles_trimmed_audition_into_dense_training_tracks(tmp_path: Path) -> None:
    sample_rate_hz = 8_000
    rendered = []
    for index, utterance in enumerate(audition_utterances()):
        leading = np.zeros(round(0.12 * sample_rate_hz), dtype=np.float32)
        speech = np.full(round((0.6 + index * 0.1) * sample_rate_hz), 0.1, dtype=np.float32)
        trailing = np.zeros(round(0.16 * sample_rate_hz), dtype=np.float32)
        rendered.append(
            materialize_trimmed_audition_utterance(
                candidate_id="woman_easy_1",
                utterance=utterance,
                samples=np.concatenate((leading, speech, trailing)),
                sample_rate_hz=sample_rate_hz,
                generation_seconds=0.1,
                output_path=tmp_path / "clips" / f"{utterance.utterance_id}.wav",
            )
        )
    compiled = compile_audition_candidate_training_samples(
        candidate_id="woman_easy_1",
        rendered_utterances=tuple(rendered),
        output_directory=tmp_path / "compiled",
        config=ConversationCompilerConfig(
            crop_variant_count=4,
            assistant_only_fraction=0.0,
            user_only_fraction=0.0,
            event_light_fraction=0.0,
            assistant_duration_variation=0.0,
        ),
    )

    assert all(
        isinstance(turn.timing, AfterUserAssistantTiming) for turn in compiled.plan.assistant_turns
    )
    assert all(len(crop.labels.p_user_floor_now) == 250 for crop in compiled.crops)
    assert any(1.0 in crop.labels.turn_completion for crop in compiled.crops)
    assert any(1.0 in crop.labels.non_floor_feedback for crop in compiled.crops)
    assert any(1.0 in crop.labels.floor_take for crop in compiled.crops)
    interruption_event = next(
        event for event in compiled.crops[0].user_events if event.event_id == "interruption"
    )
    interruption_crop = next(
        crop
        for crop in compiled.crops
        if crop.source_start_seconds - crop.left_padding_seconds
        <= interruption_event.start_seconds
        < crop.source_start_seconds - crop.left_padding_seconds + crop.duration_seconds
    )
    crop_origin = interruption_crop.source_start_seconds - interruption_crop.left_padding_seconds
    interruption_frame = round((interruption_event.start_seconds - crop_origin) / 0.08 - 0.5)
    assert interruption_crop.labels.assistant_speaking_probability[interruption_frame] > 0.8
    materialized = materialize_crop_audio(
        compiled,
        interruption_crop,
        tmp_path / "materialized.wav",
    )
    assert materialized.stat().st_size > 44
