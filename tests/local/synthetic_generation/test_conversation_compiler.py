from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np

from app.local.synthetic_generation.audio_files import read_mono_pcm16_audio
from app.local.synthetic_generation.conversation_compiler import (
    AfterAssistantUserTiming,
    AfterUserAssistantTiming,
    AssistantProbabilityDip,
    CompletionPlacement,
    ConversationCompilerConfig,
    ConversationCompositionPlan,
    CropSamplingStratum,
    DuringAssistantUserTiming,
    FixedAssistantTiming,
    FixedUserTiming,
    HoldPlacement,
    InterruptionFloorClaimPlacement,
    MeasuredSilence,
    NonFloorFeedbackPlacement,
    PreservePlannedDuration,
    RenderedUserClip,
    ResponseFloorClaimPlacement,
    VirtualAssistantTurn,
    compile_conversation,
    materialize_crop_audio,
    sampling_stratum_for_index,
)
from app.local.synthetic_generation.conversation_pipeline import summarize_crop_sampling


def test_compile_conversation_composes_audio_and_all_dense_labels(tmp_path: Path) -> None:
    clips = (
        _clip(tmp_path, "opening", 2.0, 0.1, 1.8),
        _clip(tmp_path, "feedback", 0.5, 0.05, 0.42),
        _clip(tmp_path, "response", 2.0, 0.1, 1.9),
        _clip(tmp_path, "interruption", 2.0, 0.05, 1.9),
        _clip(
            tmp_path,
            "hold_turn",
            4.0,
            0.1,
            3.8,
            continuation_silences=(MeasuredSilence(start_seconds=1.2, end_seconds=1.9),),
        ),
    )
    plan = ConversationCompositionPlan(
        conversation_id="multi_event",
        seed=17,
        duration_seconds=40.0,
        user_events=(
            CompletionPlacement(
                event_id="opening",
                clip_id="opening",
                timing=FixedUserTiming(start_seconds=1.0),
            ),
            NonFloorFeedbackPlacement(
                event_id="feedback",
                clip_id="feedback",
                timing=DuringAssistantUserTiming(
                    assistant_turn_id="assistant_one", position_fraction=0.4
                ),
            ),
            ResponseFloorClaimPlacement(
                event_id="response",
                clip_id="response",
                timing=AfterAssistantUserTiming(
                    assistant_turn_id="assistant_one", delay_seconds=0.5
                ),
            ),
            InterruptionFloorClaimPlacement(
                event_id="interruption",
                clip_id="interruption",
                timing=DuringAssistantUserTiming(
                    assistant_turn_id="assistant_two", position_fraction=0.7
                ),
            ),
            HoldPlacement(
                event_id="hold_turn",
                clip_id="hold_turn",
                timing=FixedUserTiming(start_seconds=20.0),
            ),
        ),
        assistant_turns=(
            VirtualAssistantTurn(
                turn_id="assistant_one",
                timing=AfterUserAssistantTiming(user_event_id="opening", delay_seconds=0.2),
                duration_seconds=5.0,
                probability_dips=(
                    AssistantProbabilityDip(
                        start_offset_seconds=1.7,
                        end_offset_seconds=2.4,
                        probability=0.7,
                    ),
                ),
            ),
            VirtualAssistantTurn(
                turn_id="assistant_two",
                timing=AfterUserAssistantTiming(user_event_id="response", delay_seconds=1.5),
                duration_seconds=3.4,
            ),
        ),
    )
    config = ConversationCompilerConfig(
        crop_variant_count=5,
        assistant_only_fraction=0.0,
        user_only_fraction=0.0,
        event_light_fraction=0.0,
        assistant_duration_variation=0.15,
    )

    compiled = compile_conversation(plan, clips, tmp_path / "conversation.wav", config)

    assert _wave_duration(compiled.audio_path) == 25.0
    assert compiled.plan.duration_seconds == 25.0
    assert len(compiled.crops) == 5
    assert all(len(crop.labels.p_user_floor_now) == 250 for crop in compiled.crops)
    assert all(len(crop.labels.speculative_eot) == 4 for crop in compiled.crops)
    for crop in compiled.crops:
        for frame_values in zip(
            *(track.probabilities for track in crop.labels.speculative_eot), strict=True
        ):
            visible = tuple(value for value in frame_values if value != -1.0)
            assert visible == tuple(sorted(visible))
    crop_audio_path = materialize_crop_audio(compiled, compiled.crops[0], tmp_path / "crop.wav")
    assert _wave_duration(crop_audio_path) == 20.0
    assert (
        sum(value == 1.0 for crop in compiled.crops for value in crop.labels.turn_completion) >= 4
    )
    assert any(value == 1.0 for crop in compiled.crops for value in crop.labels.continuation_pause)
    assert (
        max(
            sum(value == 1.0 for value in crop.labels.continuation_pause) for crop in compiled.crops
        )
        >= 8
    )
    assert any(value == 1.0 for crop in compiled.crops for value in crop.labels.non_floor_feedback)
    assert (
        max(
            sum(value == 1.0 for value in crop.labels.non_floor_feedback) for crop in compiled.crops
        )
        >= 4
    )

    assert any(value == 1.0 for crop in compiled.crops for value in crop.labels.floor_take)
    assert any(
        0.0 < value < 0.9
        for crop in compiled.crops
        for value in crop.labels.assistant_speaking_probability
    )
    assert any(
        value == 1.0
        for crop in compiled.crops
        for horizon in crop.labels.speculative_eot
        for value in horizon.probabilities
    )
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


def test_fitted_source_duration_accepts_rendered_timeline_longer_than_estimate(
    tmp_path: Path,
) -> None:
    clip = _clip(tmp_path, "longer_than_estimate", 12.0, 0.0, 12.0)
    plan = ConversationCompositionPlan(
        conversation_id="longer_than_estimate",
        seed=21,
        duration_seconds=8.0,
        user_events=(
            CompletionPlacement(
                event_id="longer_than_estimate",
                clip_id="longer_than_estimate",
                timing=FixedUserTiming(start_seconds=0.0),
            ),
        ),
    )

    compiled = compile_conversation(
        plan,
        (clip,),
        tmp_path / "longer-than-estimate.wav",
        ConversationCompilerConfig(
            crop_variant_count=1,
            assistant_only_fraction=0.0,
            user_only_fraction=0.0,
            event_light_fraction=0.0,
        ),
    )

    assert compiled.plan.duration_seconds == 13.0


def test_compile_conversation_supports_assistant_only_event_light_crops(tmp_path: Path) -> None:
    clip = _clip(tmp_path, "assistant_overlap", 20.0, 0.0, 20.0)
    plan = ConversationCompositionPlan(
        conversation_id="assistant_only",
        seed=9,
        duration_seconds=20.0,
        user_events=(
            CompletionPlacement(
                event_id="assistant_overlap",
                clip_id="assistant_overlap",
                timing=FixedUserTiming(start_seconds=0.0),
            ),
        ),
        assistant_turns=(
            VirtualAssistantTurn(
                turn_id="long_assistant",
                timing=FixedAssistantTiming(start_seconds=0.0),
                duration_seconds=19.0,
            ),
        ),
    )

    compiled = compile_conversation(
        plan,
        (clip,),
        tmp_path / "assistant-only.wav",
        ConversationCompilerConfig(
            crop_variant_count=2,
            assistant_only_fraction=1.0,
            user_only_fraction=0.0,
            event_light_fraction=0.0,
        ),
    )

    assert all(crop.event_light for crop in compiled.crops)
    assert all(value == -1.0 for value in compiled.crops[0].labels.turn_completion)
    assert max(compiled.crops[0].labels.assistant_speaking_probability) > 0.9
    assert set(compiled.crops[0].labels.p_user_floor_now) == {0.0}
    assert compiled.crops[0].user_events == ()
    crop_path = materialize_crop_audio(compiled, compiled.crops[0], tmp_path / "control.wav")
    crop_samples, _ = read_mono_pcm16_audio(crop_path)
    assert not np.any(crop_samples)


def test_compile_conversation_is_deterministic_across_epochs(tmp_path: Path) -> None:
    clip = _clip(tmp_path, "turn", 2.0, 0.1, 1.8)
    plan = ConversationCompositionPlan(
        conversation_id="deterministic",
        seed=51,
        duration_seconds=22.0,
        user_events=(
            CompletionPlacement(
                event_id="turn",
                clip_id="turn",
                timing=FixedUserTiming(start_seconds=10.0),
            ),
        ),
        assistant_turns=(
            VirtualAssistantTurn(
                turn_id="assistant",
                timing=AfterUserAssistantTiming(user_event_id="turn", delay_seconds=0.5),
                duration_seconds=7.5,
            ),
        ),
    )
    config = ConversationCompilerConfig(
        crop_variant_count=4,
        assistant_only_fraction=0.0,
        user_only_fraction=0.0,
        event_light_fraction=0.0,
    )

    first = compile_conversation(plan, (clip,), tmp_path / "first.wav", config)
    second = compile_conversation(plan, (clip,), tmp_path / "second.wav", config)

    assert tuple(crop.crop_id for crop in first.crops) == tuple(
        crop.crop_id for crop in second.crops
    )
    assert tuple(crop.source_start_seconds for crop in first.crops) == tuple(
        crop.source_start_seconds for crop in second.crops
    )
    assert len(set(crop.source_start_seconds for crop in first.crops)) > 1
    assert tuple(crop.assistant_duration_scale for crop in first.crops) == tuple(
        crop.assistant_duration_scale for crop in second.crops
    )


def test_sampling_strata_follow_corpus_level_fractions() -> None:
    config = ConversationCompilerConfig(
        crop_variant_count=4,
        assistant_only_fraction=0.1,
        user_only_fraction=0.1,
        event_light_fraction=0.1,
    )

    strata = tuple(sampling_stratum_for_index(index, config) for index in range(1_000))

    expected = {
        CropSamplingStratum.ASSISTANT_ONLY: 100,
        CropSamplingStratum.USER_ONLY: 100,
        CropSamplingStratum.EVENT_LIGHT: 100,
        CropSamplingStratum.EVENT_FOCUSED: 700,
    }
    assert all(abs(strata.count(stratum) - count) <= 2 for stratum, count in expected.items())


def test_assistant_duration_variation_reflows_audio_and_eot_labels(tmp_path: Path) -> None:
    opening = _clip(tmp_path, "reflow_opening", 1.0, 0.0, 0.9)
    response = _clip(tmp_path, "reflow_response", 1.0, 0.1, 0.9)
    plan = ConversationCompositionPlan(
        conversation_id="reflow",
        seed=3,
        duration_seconds=12.0,
        user_events=(
            CompletionPlacement(
                event_id="opening",
                clip_id="reflow_opening",
                timing=FixedUserTiming(start_seconds=0.5),
            ),
            ResponseFloorClaimPlacement(
                event_id="response",
                clip_id="reflow_response",
                timing=AfterAssistantUserTiming(assistant_turn_id="assistant", delay_seconds=0.2),
            ),
        ),
        assistant_turns=(
            VirtualAssistantTurn(
                turn_id="assistant",
                timing=AfterUserAssistantTiming(user_event_id="opening", delay_seconds=0.1),
                duration_seconds=4.0,
            ),
        ),
    )
    compiled = compile_conversation(
        plan,
        (opening, response),
        tmp_path / "reflow-base.wav",
        ConversationCompilerConfig(
            crop_variant_count=6,
            assistant_only_fraction=0.0,
            user_only_fraction=0.0,
            event_light_fraction=0.0,
            assistant_duration_variation=0.25,
        ),
    )
    early = min(compiled.crops, key=lambda crop: crop.assistant_duration_scale)
    late = max(compiled.crops, key=lambda crop: crop.assistant_duration_scale)
    early_response = next(event for event in early.user_events if event.event_id == "response")
    late_response = next(event for event in late.user_events if event.event_id == "response")

    expected_shift = (late.assistant_duration_scale - early.assistant_duration_scale) * 4.0
    assert np.isclose(late_response.start_seconds - early_response.start_seconds, expected_shift)
    for crop, response_event in ((early, early_response), (late, late_response)):
        crop_origin = crop.source_start_seconds - crop.left_padding_seconds
        expected_eot = response_event.start_seconds + response.active_end_seconds
        labeled_eot_times = tuple(
            crop_origin + (index + 0.5) * 0.08
            for index, target in enumerate(crop.labels.turn_completion)
            if target == 1.0
        )
        assert min(abs(time_seconds - expected_eot) for time_seconds in labeled_eot_times) < 0.08
        crop_path = materialize_crop_audio(
            compiled, crop, tmp_path / f"reflow-{crop.variant_index}.wav"
        )
        audio_onset = _activity_onset_near(
            crop_path,
            response_event.start_seconds - crop_origin,
        )
        expected_onset = response_event.start_seconds - crop_origin + response.active_start_seconds
        assert abs(audio_onset - expected_onset) < 0.01


def test_composition_plan_allows_rendered_reflow_beyond_prompt_target_limit() -> None:
    plan = ConversationCompositionPlan(
        conversation_id="long_rendered_reflow",
        seed=9,
        duration_seconds=150.0,
        user_events=(),
    )

    assert plan.duration_seconds == 150.0


def test_subframe_feedback_falls_back_to_event_light_crop(tmp_path: Path) -> None:
    feedback = _clip(tmp_path, "subframe_feedback", 0.02, 0.0, 0.02)
    plan = ConversationCompositionPlan(
        conversation_id="subframe_feedback",
        seed=11,
        duration_seconds=20.0,
        user_events=(
            NonFloorFeedbackPlacement(
                event_id="feedback",
                clip_id="subframe_feedback",
                timing=DuringAssistantUserTiming(
                    assistant_turn_id="assistant",
                    position_fraction=0.513,
                ),
            ),
        ),
        assistant_turns=(
            VirtualAssistantTurn(
                turn_id="assistant",
                timing=FixedAssistantTiming(start_seconds=0.0),
                duration_seconds=10.0,
            ),
        ),
    )

    compiled = compile_conversation(
        plan,
        (feedback,),
        tmp_path / "subframe-feedback.wav",
        ConversationCompilerConfig(
            crop_variant_count=1,
            assistant_only_fraction=0.0,
            user_only_fraction=0.0,
            event_light_fraction=0.0,
            assistant_duration_variation=0.0,
        ),
    )

    assert compiled.crops[0].sampling_stratum is CropSamplingStratum.EVENT_LIGHT
    assert compiled.crops[0].event_light


def test_long_source_materializes_every_sampling_control_without_padding(tmp_path: Path) -> None:
    extended = _clip(tmp_path, "extended", 26.0, 0.0, 26.0)
    closing = _clip(tmp_path, "closing", 1.0, 0.0, 0.9)
    plan = ConversationCompositionPlan(
        conversation_id="long_controls",
        seed=101,
        duration_seconds=100.0,
        user_events=(
            CompletionPlacement(
                event_id="extended",
                clip_id="extended",
                timing=FixedUserTiming(start_seconds=5.0),
            ),
            CompletionPlacement(
                event_id="closing",
                clip_id="closing",
                timing=FixedUserTiming(start_seconds=75.0),
            ),
        ),
        assistant_turns=(
            VirtualAssistantTurn(
                turn_id="long_assistant",
                timing=FixedAssistantTiming(start_seconds=0.0),
                duration_seconds=100.0,
            ),
        ),
    )

    compiled = compile_conversation(
        plan,
        (extended, closing),
        tmp_path / "long-controls.wav",
        ConversationCompilerConfig(
            crop_variant_count=8,
            assistant_only_fraction=0.125,
            user_only_fraction=0.125,
            event_light_fraction=0.125,
            assistant_duration_variation=0.0,
            source_duration=PreservePlannedDuration(),
        ),
    )

    assert _wave_duration(compiled.audio_path) == 100.0
    assert not any(crop.padded for crop in compiled.crops)
    assert {crop.sampling_stratum for crop in compiled.crops} == set(CropSamplingStratum)
    assistant_crop = next(
        crop
        for crop in compiled.crops
        if crop.sampling_stratum is CropSamplingStratum.ASSISTANT_ONLY
    )
    user_crop = next(
        crop for crop in compiled.crops if crop.sampling_stratum is CropSamplingStratum.USER_ONLY
    )
    event_light_crop = next(
        crop for crop in compiled.crops if crop.sampling_stratum is CropSamplingStratum.EVENT_LIGHT
    )
    assert assistant_crop.assistant_only
    assert user_crop.user_only
    assert assistant_crop.user_events == ()
    assert user_crop.assistant_turns == ()
    assert event_light_crop.event_light
    assert all(
        value != 1.0
        for track in (
            event_light_crop.labels.turn_completion,
            event_light_crop.labels.continuation_pause,
            event_light_crop.labels.non_floor_feedback,
            event_light_crop.labels.floor_take,
        )
        for value in track
    )
    summary = summarize_crop_sampling(compiled.crops)
    assert summary.assistant_only_fraction >= 0.1
    assert summary.user_only_fraction >= 0.1
    assert summary.event_light_fraction >= 0.1
    assert summary.control_quotas_satisfied
    assert summary.padding_limit_satisfied


def test_short_source_reports_last_resort_padding(tmp_path: Path) -> None:
    clip = _clip(tmp_path, "short", 2.0, 0.0, 1.8)
    plan = ConversationCompositionPlan(
        conversation_id="short_padding",
        seed=19,
        duration_seconds=12.0,
        user_events=(
            CompletionPlacement(
                event_id="short",
                clip_id="short",
                timing=FixedUserTiming(start_seconds=5.0),
            ),
        ),
    )

    compiled = compile_conversation(
        plan,
        (clip,),
        tmp_path / "short-padding.wav",
        ConversationCompilerConfig(
            crop_variant_count=1,
            assistant_only_fraction=0.0,
            user_only_fraction=0.0,
            event_light_fraction=0.0,
            source_duration=PreservePlannedDuration(),
        ),
    )

    assert compiled.crops[0].padded
    assert np.isclose(
        compiled.crops[0].left_padding_seconds + compiled.crops[0].right_padding_seconds,
        8.0,
    )


def _clip(
    directory: Path,
    clip_id: str,
    duration_seconds: float,
    active_start_seconds: float,
    active_end_seconds: float,
    continuation_silences: tuple[MeasuredSilence, ...] = (),
) -> RenderedUserClip:
    path = directory / f"{clip_id}.wav"
    _write_tone(
        path,
        duration_seconds,
        active_start_seconds,
        active_end_seconds,
        continuation_silences,
    )
    return RenderedUserClip(
        clip_id=clip_id,
        audio_path=path,
        audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        duration_seconds=duration_seconds,
        active_start_seconds=active_start_seconds,
        active_end_seconds=active_end_seconds,
        continuation_silences=continuation_silences,
    )


def _write_tone(
    path: Path,
    duration_seconds: float,
    active_start_seconds: float,
    active_end_seconds: float,
    continuation_silences: tuple[MeasuredSilence, ...],
) -> None:
    sample_rate_hz = 16_000
    times = np.arange(round(sample_rate_hz * duration_seconds)) / sample_rate_hz
    active = (times >= active_start_seconds) & (times < active_end_seconds)
    for silence in continuation_silences:
        active &= (times < silence.start_seconds) | (times >= silence.end_seconds)
    waveform = np.sin(2.0 * np.pi * 220.0 * times) * 0.1 * active
    samples = (waveform * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate_hz)
        audio_file.writeframes(samples.tobytes())


def _wave_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio_file:
        return audio_file.getnframes() / audio_file.getframerate()


def _activity_onset_near(path: Path, expected_clip_start_seconds: float) -> float:
    with wave.open(str(path), "rb") as audio_file:
        sample_rate_hz = audio_file.getframerate()
        samples = np.frombuffer(audio_file.readframes(audio_file.getnframes()), dtype="<i2")
    search_start = max(0, round((expected_clip_start_seconds - 0.02) * sample_rate_hz))
    search_end = min(samples.size, round((expected_clip_start_seconds + 0.3) * sample_rate_hz))
    active_indices = np.flatnonzero(np.abs(samples[search_start:search_end]) > 100)
    assert active_indices.size > 0
    return (search_start + int(active_indices[0])) / sample_rate_hz
