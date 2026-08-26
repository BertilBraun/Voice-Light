from __future__ import annotations

import hashlib
import wave
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from app.local.synthetic_generation.models import SyntheticModel
from app.local.training_samples.constants import FRAME_SECONDS, INPUT_DURATION_SECONDS

MASKED_TARGET = -1.0


class MeasuredSilence(SyntheticModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_interval(self) -> MeasuredSilence:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Measured silence end must follow its start.")
        return self


class RenderedUserClip(SyntheticModel):
    clip_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(gt=0.0)
    active_start_seconds: float = Field(ge=0.0)
    active_end_seconds: float = Field(gt=0.0)
    continuation_silences: tuple[MeasuredSilence, ...] = ()

    @model_validator(mode="after")
    def validate_measurements(self) -> RenderedUserClip:
        if self.active_end_seconds > self.duration_seconds:
            raise ValueError("Clip active end exceeds its measured duration.")
        if self.active_start_seconds >= self.active_end_seconds:
            raise ValueError("Clip active start must precede its active end.")
        for silence in self.continuation_silences:
            if not self.active_start_seconds < silence.start_seconds:
                raise ValueError("Continuation silence must follow active speech onset.")
            if silence.end_seconds >= self.active_end_seconds:
                raise ValueError("Continuation silence must be internal to active speech.")
        return self


class FixedUserTiming(SyntheticModel):
    kind: Literal["fixed"] = "fixed"
    start_seconds: float = Field(ge=0.0)


class AfterAssistantUserTiming(SyntheticModel):
    kind: Literal["after_assistant"] = "after_assistant"
    assistant_turn_id: str
    delay_seconds: float = Field(ge=0.0)


class AfterUserUserTiming(SyntheticModel):
    kind: Literal["after_user"] = "after_user"
    user_event_id: str
    delay_seconds: float = Field(ge=0.0)


class DuringAssistantUserTiming(SyntheticModel):
    kind: Literal["during_assistant"] = "during_assistant"
    assistant_turn_id: str
    position_fraction: float = Field(gt=0.0, lt=1.0)


FloorOwningUserTiming = Annotated[
    FixedUserTiming | AfterAssistantUserTiming | AfterUserUserTiming,
    Field(discriminator="kind"),
]


class CompletionPlacement(SyntheticModel):
    kind: Literal["completion"] = "completion"
    event_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    clip_id: str
    timing: FloorOwningUserTiming


class HoldPlacement(SyntheticModel):
    kind: Literal["hold"] = "hold"
    event_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    clip_id: str
    timing: FloorOwningUserTiming


class NonFloorFeedbackPlacement(SyntheticModel):
    kind: Literal["non_floor_feedback"] = "non_floor_feedback"
    event_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    clip_id: str
    timing: DuringAssistantUserTiming


class ResponseFloorClaimPlacement(SyntheticModel):
    kind: Literal["response_floor_claim"] = "response_floor_claim"
    event_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    clip_id: str
    timing: AfterAssistantUserTiming


class InterruptionFloorClaimPlacement(SyntheticModel):
    kind: Literal["interruption_floor_claim"] = "interruption_floor_claim"
    event_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    clip_id: str
    timing: DuringAssistantUserTiming
    assistant_yield_delay_seconds: float = Field(default=0.25, ge=0.08, le=0.6)


UserClipPlacement = Annotated[
    CompletionPlacement
    | HoldPlacement
    | NonFloorFeedbackPlacement
    | ResponseFloorClaimPlacement
    | InterruptionFloorClaimPlacement,
    Field(discriminator="kind"),
]


class AssistantProbabilityDip(SyntheticModel):
    start_offset_seconds: float = Field(ge=0.0)
    end_offset_seconds: float = Field(gt=0.0)
    probability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> AssistantProbabilityDip:
        if self.end_offset_seconds <= self.start_offset_seconds:
            raise ValueError("Assistant probability dip end must follow its start.")
        return self


class FixedAssistantTiming(SyntheticModel):
    kind: Literal["fixed"] = "fixed"
    start_seconds: float = Field(ge=0.0)


class AfterUserAssistantTiming(SyntheticModel):
    kind: Literal["after_user"] = "after_user"
    user_event_id: str
    delay_seconds: float = Field(ge=0.0)


AssistantPlacementTiming = Annotated[
    FixedAssistantTiming | AfterUserAssistantTiming,
    Field(discriminator="kind"),
]


class VirtualAssistantTurn(SyntheticModel):
    turn_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    timing: AssistantPlacementTiming
    duration_seconds: float = Field(gt=0.0)
    speaking_probability: float = Field(default=0.95, ge=0.0, le=1.0)
    transition_seconds: float = Field(default=0.16, ge=0.0, le=1.0)
    probability_dips: tuple[AssistantProbabilityDip, ...] = ()

    @model_validator(mode="after")
    def validate_dips(self) -> VirtualAssistantTurn:
        for probability_dip in self.probability_dips:
            if probability_dip.end_offset_seconds > self.duration_seconds:
                raise ValueError("Assistant probability dip ends after its turn.")
        return self


class ConversationCompositionPlan(SyntheticModel):
    schema_version: Literal["voice-light-conversation-composition-v1"] = (
        "voice-light-conversation-composition-v1"
    )
    conversation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    seed: int = Field(ge=0)
    duration_seconds: float = Field(gt=0.0, le=120.0)
    user_events: tuple[UserClipPlacement, ...]
    assistant_turns: tuple[VirtualAssistantTurn, ...] = ()

    @model_validator(mode="after")
    def validate_timeline(self) -> ConversationCompositionPlan:
        event_ids = tuple(event.event_id for event in self.user_events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("User event IDs must be unique.")
        turn_ids = tuple(turn.turn_id for turn in self.assistant_turns)
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("Virtual assistant turn IDs must be unique.")
        return self


class ConversationCompilerConfig(SyntheticModel):
    sample_rate_hz: int = Field(default=16_000, gt=0)
    crop_duration_seconds: float = Field(default=INPUT_DURATION_SECONDS, gt=0.0)
    crop_variant_count: int = Field(default=8, gt=0)
    assistant_only_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    user_only_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    event_light_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    assistant_duration_variation: float = Field(default=0.1, ge=0.0, le=0.4)
    speculative_eot_horizons_seconds: tuple[float, ...] = (0.5, 1.0)

    @model_validator(mode="after")
    def validate_frame_contract(self) -> ConversationCompilerConfig:
        frame_count = self.crop_duration_seconds / FRAME_SECONDS
        if not frame_count.is_integer():
            raise ValueError("Crop duration must contain a whole number of label frames.")
        if not self.speculative_eot_horizons_seconds:
            raise ValueError("At least one speculative EOT horizon is required.")
        if any(horizon <= 0.0 for horizon in self.speculative_eot_horizons_seconds):
            raise ValueError("Speculative EOT horizons must be positive.")
        requested_control_count = sum(
            _fraction_count(self.crop_variant_count, fraction)
            for fraction in (
                self.assistant_only_fraction,
                self.user_only_fraction,
                self.event_light_fraction,
            )
        )
        if requested_control_count > self.crop_variant_count:
            raise ValueError("Requested crop control strata exceed the crop variant count.")
        return self


class CropSamplingStratum(StrEnum):
    EVENT_FOCUSED = "event_focused"
    ASSISTANT_ONLY = "assistant_only"
    USER_ONLY = "user_only"
    EVENT_LIGHT = "event_light"


class SpeculativeEotTrack(SyntheticModel):
    horizon_seconds: float = Field(gt=0.0)
    probabilities: tuple[float, ...]


class ResolvedUserEvent(SyntheticModel):
    event_id: str
    clip_id: str
    kind: Literal[
        "completion",
        "hold",
        "non_floor_feedback",
        "response_floor_claim",
        "interruption_floor_claim",
    ]
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)


class ResolvedAssistantTurn(SyntheticModel):
    turn_id: str
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    speaking_probability: float = Field(ge=0.0, le=1.0)
    transition_seconds: float = Field(ge=0.0)
    probability_dips: tuple[AssistantProbabilityDip, ...]


class ConversationFrameTracks(SyntheticModel):
    frame_seconds: float = FRAME_SECONDS
    assistant_speaking_probability: tuple[float, ...]
    p_user_floor_now: tuple[float, ...]
    turn_completion: tuple[float, ...]
    continuation_pause: tuple[float, ...]
    non_floor_feedback: tuple[float, ...]
    floor_take: tuple[float, ...]
    speculative_eot: tuple[SpeculativeEotTrack, ...]

    @model_validator(mode="after")
    def validate_tracks(self) -> ConversationFrameTracks:
        frame_count = len(self.assistant_speaking_probability)
        tracks = (
            self.p_user_floor_now,
            self.turn_completion,
            self.continuation_pause,
            self.non_floor_feedback,
            self.floor_take,
            *(track.probabilities for track in self.speculative_eot),
        )
        if any(len(track) != frame_count for track in tracks):
            raise ValueError("All compiled conversation tracks must have equal frame counts.")
        if any(
            not 0.0 <= probability <= 1.0 for probability in self.assistant_speaking_probability
        ):
            raise ValueError("Assistant-speaking input must contain probabilities.")
        if any(
            value != MASKED_TARGET and not 0.0 <= value <= 1.0
            for track in tracks
            for value in track
        ):
            raise ValueError("Training targets must be probabilities or the mask sentinel.")
        return self


class TrainingCropPlan(SyntheticModel):
    crop_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant_index: int = Field(ge=0)
    source_start_seconds: float = Field(ge=0.0)
    source_end_seconds: float = Field(ge=0.0)
    left_padding_seconds: float = Field(ge=0.0)
    right_padding_seconds: float = Field(ge=0.0)
    duration_seconds: float = Field(gt=0.0)
    assistant_duration_scale: float = Field(gt=0.0)
    sampling_stratum: CropSamplingStratum
    assistant_only: bool
    user_only: bool
    event_light: bool
    padded: bool
    user_events: tuple[ResolvedUserEvent, ...]
    assistant_turns: tuple[ResolvedAssistantTurn, ...]
    labels: ConversationFrameTracks

    @model_validator(mode="after")
    def validate_duration(self) -> TrainingCropPlan:
        represented_seconds = (
            self.left_padding_seconds
            + self.source_end_seconds
            - self.source_start_seconds
            + self.right_padding_seconds
        )
        if not np.isclose(represented_seconds, self.duration_seconds, atol=1e-6):
            raise ValueError("Crop source and padding do not equal its duration.")
        expected_frames = round(self.duration_seconds / self.labels.frame_seconds)
        if len(self.labels.p_user_floor_now) != expected_frames:
            raise ValueError("Crop label count does not match its duration.")
        has_padding = self.left_padding_seconds > 0.0 or self.right_padding_seconds > 0.0
        if self.padded != has_padding:
            raise ValueError("Crop padding status does not match its source bounds.")
        match self.sampling_stratum:
            case CropSamplingStratum.EVENT_FOCUSED if self.event_light:
                raise ValueError("Event-focused crops must contain a supervised event.")
            case CropSamplingStratum.ASSISTANT_ONLY if not self.assistant_only:
                raise ValueError("Assistant-only crops must satisfy the requested control.")
            case CropSamplingStratum.USER_ONLY if not self.user_only:
                raise ValueError("User-only crops must satisfy the requested control.")
            case CropSamplingStratum.EVENT_LIGHT if not self.event_light:
                raise ValueError("Event-light crops must contain no supervised event.")
            case _:
                pass
        return self


class CompiledConversation(SyntheticModel):
    schema_version: Literal["voice-light-compiled-conversation-v1"] = (
        "voice-light-compiled-conversation-v1"
    )
    plan: ConversationCompositionPlan
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: int
    rendered_clips: tuple[RenderedUserClip, ...]
    crops: tuple[TrainingCropPlan, ...]


def compile_conversation(
    plan: ConversationCompositionPlan,
    rendered_clips: tuple[RenderedUserClip, ...],
    output_audio_path: Path,
    config: ConversationCompilerConfig,
) -> CompiledConversation:
    clips_by_id = _validated_clips(plan, rendered_clips)
    base_user_events, _ = _resolve_variant_timeline(plan, clips_by_id, 1.0)
    samples = _compose_user_waveform(
        plan.duration_seconds, base_user_events, clips_by_id, config.sample_rate_hz
    )
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm16_wave(output_audio_path, samples, config.sample_rate_hz)
    crops = tuple(
        _compile_crop(plan, clips_by_id, variant_index, config)
        for variant_index in range(config.crop_variant_count)
    )
    return CompiledConversation(
        plan=plan,
        audio_path=output_audio_path,
        audio_sha256=_file_sha256(output_audio_path),
        sample_rate_hz=config.sample_rate_hz,
        rendered_clips=rendered_clips,
        crops=crops,
    )


def materialize_crop_audio(
    conversation: CompiledConversation,
    crop: TrainingCropPlan,
    output_audio_path: Path,
) -> Path:
    clips_by_id = {clip.clip_id: clip for clip in conversation.rendered_clips}
    variant_duration = max(
        conversation.plan.duration_seconds,
        *(event.end_seconds for event in crop.user_events),
        *(turn.end_seconds for turn in crop.assistant_turns),
    )
    source_samples = _compose_user_waveform(
        variant_duration, crop.user_events, clips_by_id, conversation.sample_rate_hz
    )
    output_count = round(crop.duration_seconds * conversation.sample_rate_hz)
    output = np.zeros(output_count, dtype=np.float32)
    source_start = round(crop.source_start_seconds * conversation.sample_rate_hz)
    source_end = round(crop.source_end_seconds * conversation.sample_rate_hz)
    output_start = round(crop.left_padding_seconds * conversation.sample_rate_hz)
    selected = source_samples[source_start:source_end]
    output[output_start : output_start + selected.size] = selected
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm16_wave(output_audio_path, output, conversation.sample_rate_hz)
    return output_audio_path


def _validated_clips(
    plan: ConversationCompositionPlan,
    rendered_clips: tuple[RenderedUserClip, ...],
) -> dict[str, RenderedUserClip]:
    clips_by_id = {clip.clip_id: clip for clip in rendered_clips}
    if len(clips_by_id) != len(rendered_clips):
        raise ValueError("Rendered clip IDs must be unique.")
    expected_clip_ids = {event.clip_id for event in plan.user_events}
    if expected_clip_ids != set(clips_by_id):
        raise ValueError("Rendered clips must exactly match conversation user events.")
    for event in plan.user_events:
        clip = clips_by_id[event.clip_id]
        if _file_sha256(clip.audio_path) != clip.audio_sha256:
            raise ValueError(f"Rendered clip hash changed for {clip.clip_id}.")
        clip_samples, sample_rate_hz = _read_pcm16_wave(clip.audio_path)
        actual_duration_seconds = clip_samples.size / sample_rate_hz
        if not np.isclose(actual_duration_seconds, clip.duration_seconds, atol=1 / sample_rate_hz):
            raise ValueError(f"Measured duration changed for rendered clip {clip.clip_id}.")
        if event.kind == "hold" and not clip.continuation_silences:
            raise ValueError(f"Hold event {event.event_id} has no measured continuation silence.")
    _resolve_variant_timeline(plan, clips_by_id, 1.0)
    return clips_by_id


def _compose_user_waveform(
    duration_seconds: float,
    user_events: tuple[ResolvedUserEvent, ...],
    clips_by_id: dict[str, RenderedUserClip],
    sample_rate_hz: int,
) -> np.ndarray:
    output = np.zeros(round(duration_seconds * sample_rate_hz), dtype=np.float32)
    for event in user_events:
        clip_samples, clip_sample_rate_hz = _read_pcm16_wave(clips_by_id[event.clip_id].audio_path)
        resampled = _resample(clip_samples, clip_sample_rate_hz, sample_rate_hz)
        start_index = round(event.start_seconds * sample_rate_hz)
        output[start_index : start_index + resampled.size] += resampled
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.98:
        output *= 0.98 / peak
    return output


def _compile_crop(
    plan: ConversationCompositionPlan,
    clips_by_id: dict[str, RenderedUserClip],
    variant_index: int,
    config: ConversationCompilerConfig,
) -> TrainingCropPlan:
    generator = np.random.default_rng(plan.seed + variant_index * 104_729)
    variation = config.assistant_duration_variation
    assistant_scale = float(generator.uniform(1.0 - variation, 1.0 + variation))
    user_events, assistant_turns = _resolve_variant_timeline(plan, clips_by_id, assistant_scale)
    completions, hold_intervals, feedback_intervals, floor_takes, user_floor = _semantic_timeline(
        user_events, clips_by_id
    )
    variant_duration = max(
        plan.duration_seconds,
        *(event.end_seconds for event in user_events),
        *(turn.end_seconds for turn in assistant_turns),
    )
    if variant_duration > 120.0:
        raise ValueError("Reflowed conversation exceeds the 120-second source limit.")
    feedback_starts = tuple(start for start, _ in feedback_intervals)
    hold_starts = tuple(start for start, _ in hold_intervals)
    event_times = completions + hold_starts + feedback_starts + floor_takes
    requested_sampling_stratum = _sampling_stratum(variant_index, config)
    crop_start, sampling_stratum = _crop_start_for_stratum(
        variant_index,
        variant_duration,
        event_times,
        user_events,
        assistant_turns,
        user_floor,
        requested_sampling_stratum,
        config,
        generator,
    )
    source_start = max(0.0, crop_start)
    source_end = min(variant_duration, crop_start + config.crop_duration_seconds)
    left_padding = max(0.0, -crop_start)
    right_padding = max(0.0, crop_start + config.crop_duration_seconds - variant_duration)
    frame_count = round(config.crop_duration_seconds / FRAME_SECONDS)
    frame_times = tuple(crop_start + (index + 0.5) * FRAME_SECONDS for index in range(frame_count))
    assistant_probability = tuple(
        _assistant_probability_at(time_seconds, assistant_turns)
        if 0.0 <= time_seconds < variant_duration
        else 0.0
        for time_seconds in frame_times
    )
    speculative = tuple(
        SpeculativeEotTrack(
            horizon_seconds=horizon,
            probabilities=tuple(
                _speculative_eot_target(time_seconds, horizon, completions, variant_duration)
                for time_seconds in frame_times
            ),
        )
        for horizon in config.speculative_eot_horizons_seconds
    )
    labels = ConversationFrameTracks(
        assistant_speaking_probability=assistant_probability,
        p_user_floor_now=tuple(
            _active_at(time_seconds, user_floor)
            if 0.0 <= time_seconds < variant_duration
            else MASKED_TARGET
            for time_seconds in frame_times
        ),
        turn_completion=_sparse_targets(frame_times, completions, hold_starts),
        continuation_pause=_interval_targets(frame_times, hold_intervals, completions),
        non_floor_feedback=_interval_targets(frame_times, feedback_intervals, floor_takes),
        floor_take=_sparse_targets(frame_times, floor_takes, feedback_starts),
        speculative_eot=speculative,
    )
    crop_id = hashlib.sha256(
        f"{plan.conversation_id}:{variant_index}:{crop_start:.6f}:{assistant_scale:.6f}".encode()
    ).hexdigest()
    event_light = not any(
        value == 1.0
        for track in (
            labels.turn_completion,
            labels.continuation_pause,
            labels.non_floor_feedback,
            labels.floor_take,
            *(track.probabilities for track in labels.speculative_eot),
        )
        for value in track
    )
    assistant_only = _assistant_only_control(
        crop_start,
        config.crop_duration_seconds,
        user_events,
        assistant_turns,
    )
    user_only = _user_only_control(
        crop_start,
        config.crop_duration_seconds,
        event_times,
        user_floor,
        assistant_turns,
        max(config.speculative_eot_horizons_seconds),
    )
    padded = left_padding > 0.0 or right_padding > 0.0
    return TrainingCropPlan(
        crop_id=crop_id,
        variant_index=variant_index,
        source_start_seconds=source_start,
        source_end_seconds=source_end,
        left_padding_seconds=left_padding,
        right_padding_seconds=right_padding,
        duration_seconds=config.crop_duration_seconds,
        assistant_duration_scale=assistant_scale,
        sampling_stratum=sampling_stratum,
        assistant_only=assistant_only,
        user_only=user_only,
        event_light=event_light,
        padded=padded,
        user_events=user_events,
        assistant_turns=assistant_turns,
        labels=labels,
    )


def _resolve_variant_timeline(
    plan: ConversationCompositionPlan,
    clips_by_id: dict[str, RenderedUserClip],
    assistant_duration_scale: float,
) -> tuple[tuple[ResolvedUserEvent, ...], tuple[ResolvedAssistantTurn, ...]]:
    user_by_id = {event.event_id: event for event in plan.user_events}
    assistant_by_id = {turn.turn_id: turn for turn in plan.assistant_turns}
    resolved_users: dict[str, ResolvedUserEvent] = {}
    resolved_assistants: dict[str, ResolvedAssistantTurn] = {}
    resolving: set[str] = set()

    def resolve_user(event_id: str) -> ResolvedUserEvent:
        if event_id in resolved_users:
            return resolved_users[event_id]
        dependency_key = f"user:{event_id}"
        if dependency_key in resolving:
            raise ValueError("Conversation timing dependencies contain a cycle.")
        event = user_by_id.get(event_id)
        if event is None:
            raise ValueError(f"Unknown user timing dependency {event_id}.")
        resolving.add(dependency_key)
        match event.timing:
            case FixedUserTiming(start_seconds=start_seconds):
                resolved_start = start_seconds
            case AfterAssistantUserTiming(
                assistant_turn_id=assistant_turn_id, delay_seconds=delay_seconds
            ):
                resolved_start = resolve_assistant(assistant_turn_id).end_seconds + delay_seconds
            case AfterUserUserTiming(user_event_id=user_event_id, delay_seconds=delay_seconds):
                resolved_start = resolve_user(user_event_id).end_seconds + delay_seconds
            case DuringAssistantUserTiming(
                assistant_turn_id=assistant_turn_id, position_fraction=position_fraction
            ):
                assistant_turn = resolve_assistant(assistant_turn_id)
                nominal_duration = (
                    assistant_by_id[assistant_turn_id].duration_seconds * assistant_duration_scale
                )
                resolved_start = assistant_turn.start_seconds + nominal_duration * position_fraction
        clip = clips_by_id[event.clip_id]
        resolved = ResolvedUserEvent(
            event_id=event.event_id,
            clip_id=event.clip_id,
            kind=event.kind,
            start_seconds=resolved_start,
            end_seconds=resolved_start + clip.duration_seconds,
        )
        resolving.remove(dependency_key)
        resolved_users[event_id] = resolved
        return resolved

    def resolve_assistant(turn_id: str) -> ResolvedAssistantTurn:
        if turn_id in resolved_assistants:
            return resolved_assistants[turn_id]
        dependency_key = f"assistant:{turn_id}"
        if dependency_key in resolving:
            raise ValueError("Conversation timing dependencies contain a cycle.")
        turn = assistant_by_id.get(turn_id)
        if turn is None:
            raise ValueError(f"Unknown assistant timing dependency {turn_id}.")
        resolving.add(dependency_key)
        match turn.timing:
            case FixedAssistantTiming(start_seconds=start_seconds):
                resolved_start = start_seconds
            case AfterUserAssistantTiming(user_event_id=user_event_id, delay_seconds=delay_seconds):
                resolved_start = resolve_user(user_event_id).end_seconds + delay_seconds
        scaled_duration = turn.duration_seconds * assistant_duration_scale
        interruption_offsets = _interruption_end_offsets(
            plan.user_events,
            turn.turn_id,
            scaled_duration,
        )
        interruption_yield_delays = _interruption_yield_delays(
            plan.user_events,
            turn.turn_id,
        )
        effective_duration = min((scaled_duration, *interruption_offsets))
        resolved = ResolvedAssistantTurn(
            turn_id=turn.turn_id,
            start_seconds=resolved_start,
            end_seconds=resolved_start + effective_duration,
            speaking_probability=turn.speaking_probability,
            transition_seconds=min((turn.transition_seconds, *interruption_yield_delays)),
            probability_dips=tuple(
                AssistantProbabilityDip(
                    start_offset_seconds=dip.start_offset_seconds * assistant_duration_scale,
                    end_offset_seconds=dip.end_offset_seconds * assistant_duration_scale,
                    probability=dip.probability,
                )
                for dip in turn.probability_dips
                if dip.start_offset_seconds * assistant_duration_scale < effective_duration
                and dip.end_offset_seconds * assistant_duration_scale <= effective_duration
            ),
        )
        resolving.remove(dependency_key)
        resolved_assistants[turn_id] = resolved
        return resolved

    users = tuple(resolve_user(event.event_id) for event in plan.user_events)
    assistants = tuple(resolve_assistant(turn.turn_id) for turn in plan.assistant_turns)
    ordered_users = sorted(users, key=lambda event: event.start_seconds)
    for first, second in zip(ordered_users, ordered_users[1:], strict=False):
        if first.end_seconds > second.start_seconds:
            raise ValueError(f"User events {first.event_id} and {second.event_id} overlap.")
    nominal_end = max(
        0.0,
        *(event.end_seconds for event in users),
        *(turn.end_seconds for turn in assistants),
    )
    if assistant_duration_scale == 1.0 and nominal_end > plan.duration_seconds:
        raise ValueError("Nominal timeline exceeds conversation duration.")
    return users, assistants


def _interruption_end_offsets(
    user_events: tuple[UserClipPlacement, ...],
    assistant_turn_id: str,
    assistant_duration_seconds: float,
) -> tuple[float, ...]:
    offsets: list[float] = []
    for event in user_events:
        match event:
            case InterruptionFloorClaimPlacement(
                timing=DuringAssistantUserTiming(
                    assistant_turn_id=referenced_turn_id,
                    position_fraction=position_fraction,
                ),
                assistant_yield_delay_seconds=yield_delay_seconds,
            ) if referenced_turn_id == assistant_turn_id:
                offsets.append(assistant_duration_seconds * position_fraction + yield_delay_seconds)
            case _:
                pass
    return tuple(offsets)


def _interruption_yield_delays(
    user_events: tuple[UserClipPlacement, ...],
    assistant_turn_id: str,
) -> tuple[float, ...]:
    delays: list[float] = []
    for event in user_events:
        match event:
            case InterruptionFloorClaimPlacement(
                timing=DuringAssistantUserTiming(assistant_turn_id=referenced_turn_id),
                assistant_yield_delay_seconds=yield_delay_seconds,
            ) if referenced_turn_id == assistant_turn_id:
                delays.append(yield_delay_seconds)
            case _:
                pass
    return tuple(delays)


def _semantic_timeline(
    user_events: tuple[ResolvedUserEvent, ...],
    clips_by_id: dict[str, RenderedUserClip],
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[tuple[float, float], ...],
]:
    completions: list[float] = []
    holds: list[tuple[float, float]] = []
    feedback: list[tuple[float, float]] = []
    floor_takes: list[float] = []
    user_floor: list[tuple[float, float]] = []
    for event in user_events:
        clip = clips_by_id[event.clip_id]
        active_start = event.start_seconds + clip.active_start_seconds
        active_end = event.start_seconds + clip.active_end_seconds
        match event.kind:
            case "completion" | "hold":
                user_floor.append((active_start, active_end))
                completions.append(active_end)
            case "response_floor_claim" | "interruption_floor_claim":
                user_floor.append((active_start, active_end))
                floor_takes.append(active_start)
                completions.append(active_end)
            case "non_floor_feedback":
                feedback.append((active_start, active_end))
        if event.kind != "non_floor_feedback":
            holds.extend(
                (
                    event.start_seconds + silence.start_seconds,
                    event.start_seconds + silence.end_seconds,
                )
                for silence in clip.continuation_silences
            )
    return (
        tuple(completions),
        tuple(holds),
        tuple(feedback),
        tuple(floor_takes),
        tuple(user_floor),
    )


def _crop_start_for_stratum(
    variant_index: int,
    duration_seconds: float,
    event_times: tuple[float, ...],
    user_events: tuple[ResolvedUserEvent, ...],
    assistant_turns: tuple[ResolvedAssistantTurn, ...],
    user_floor: tuple[tuple[float, float], ...],
    sampling_stratum: CropSamplingStratum,
    config: ConversationCompilerConfig,
    generator: np.random.Generator,
) -> tuple[float, CropSamplingStratum]:
    candidates = _candidate_crop_starts(duration_seconds, config.crop_duration_seconds)
    fallback_order = (
        sampling_stratum,
        CropSamplingStratum.EVENT_FOCUSED,
        CropSamplingStratum.EVENT_LIGHT,
    )
    for candidate_stratum in dict.fromkeys(fallback_order):
        valid = _valid_crop_starts(
            candidates,
            candidate_stratum,
            event_times,
            user_events,
            assistant_turns,
            user_floor,
            config.crop_duration_seconds,
            max(config.speculative_eot_horizons_seconds),
        )
        if valid:
            selection = int(generator.integers(0, len(valid)))
            return valid[(selection + variant_index) % len(valid)], candidate_stratum
    raise ValueError("Conversation has no valid event-focused or event-light 20-second crop.")


def _valid_crop_starts(
    candidates: tuple[float, ...],
    sampling_stratum: CropSamplingStratum,
    event_times: tuple[float, ...],
    user_events: tuple[ResolvedUserEvent, ...],
    assistant_turns: tuple[ResolvedAssistantTurn, ...],
    user_floor: tuple[tuple[float, float], ...],
    crop_duration_seconds: float,
    speculative_horizon_seconds: float,
) -> tuple[float, ...]:
    match sampling_stratum:
        case CropSamplingStratum.EVENT_FOCUSED:
            return tuple(
                start
                for start in candidates
                if _contains_event_after_context(start, crop_duration_seconds, event_times)
            )
        case CropSamplingStratum.ASSISTANT_ONLY:
            return tuple(
                start
                for start in candidates
                if _assistant_only_control(
                    start,
                    crop_duration_seconds,
                    user_events,
                    assistant_turns,
                )
            )
        case CropSamplingStratum.USER_ONLY:
            return tuple(
                start
                for start in candidates
                if _user_only_control(
                    start,
                    crop_duration_seconds,
                    event_times,
                    user_floor,
                    assistant_turns,
                    speculative_horizon_seconds,
                )
            )
        case CropSamplingStratum.EVENT_LIGHT:
            return tuple(
                start
                for start in candidates
                if not _contains_event(
                    start,
                    crop_duration_seconds,
                    event_times,
                    speculative_horizon_seconds,
                )
            )


def _sampling_stratum(
    variant_index: int,
    config: ConversationCompilerConfig,
) -> CropSamplingStratum:
    assistant_count = _fraction_count(config.crop_variant_count, config.assistant_only_fraction)
    user_count = _fraction_count(config.crop_variant_count, config.user_only_fraction)
    event_light_count = _fraction_count(config.crop_variant_count, config.event_light_fraction)
    control_count = assistant_count + user_count + event_light_count
    event_focused_count = config.crop_variant_count - control_count
    if variant_index < event_focused_count:
        return CropSamplingStratum.EVENT_FOCUSED
    if variant_index < event_focused_count + assistant_count:
        return CropSamplingStratum.ASSISTANT_ONLY
    if variant_index < event_focused_count + assistant_count + user_count:
        return CropSamplingStratum.USER_ONLY
    return CropSamplingStratum.EVENT_LIGHT


def _fraction_count(total_count: int, fraction: float) -> int:
    if fraction == 0.0:
        return 0
    return max(1, round(total_count * fraction))


def _candidate_crop_starts(
    duration_seconds: float,
    crop_duration_seconds: float,
) -> tuple[float, ...]:
    difference = duration_seconds - crop_duration_seconds
    minimum_start = min(0.0, difference)
    maximum_start = max(0.0, difference)
    frame_count = int((maximum_start - minimum_start) / FRAME_SECONDS)
    starts = tuple(minimum_start + index * FRAME_SECONDS for index in range(frame_count + 1))
    if not np.isclose(starts[-1], maximum_start, atol=1e-9):
        return (*starts, maximum_start)
    return starts


def _contains_event(
    crop_start_seconds: float,
    crop_duration_seconds: float,
    event_times: tuple[float, ...],
    future_margin_seconds: float = 0.0,
) -> bool:
    crop_end = crop_start_seconds + crop_duration_seconds + future_margin_seconds
    return any(crop_start_seconds <= event_time < crop_end for event_time in event_times)


def _contains_event_after_context(
    crop_start_seconds: float,
    crop_duration_seconds: float,
    event_times: tuple[float, ...],
) -> bool:
    supervised_start = crop_start_seconds + min(4.0, crop_duration_seconds)
    crop_end = crop_start_seconds + crop_duration_seconds
    return any(supervised_start <= event_time < crop_end for event_time in event_times)


def _assistant_only_control(
    crop_start_seconds: float,
    crop_duration_seconds: float,
    user_events: tuple[ResolvedUserEvent, ...],
    assistant_turns: tuple[ResolvedAssistantTurn, ...],
) -> bool:
    crop_end = crop_start_seconds + crop_duration_seconds
    has_user_audio = any(
        _intervals_overlap(crop_start_seconds, crop_end, event.start_seconds, event.end_seconds)
        for event in user_events
    )
    assistant_seconds = sum(
        _overlap_seconds(crop_start_seconds, crop_end, turn.start_seconds, turn.end_seconds)
        for turn in assistant_turns
    )
    return not has_user_audio and assistant_seconds >= 2.0


def _user_only_control(
    crop_start_seconds: float,
    crop_duration_seconds: float,
    event_times: tuple[float, ...],
    user_floor: tuple[tuple[float, float], ...],
    assistant_turns: tuple[ResolvedAssistantTurn, ...],
    speculative_horizon_seconds: float = 0.0,
) -> bool:
    crop_end = crop_start_seconds + crop_duration_seconds
    has_assistant_activity = any(
        _intervals_overlap(crop_start_seconds, crop_end, turn.start_seconds, turn.end_seconds)
        for turn in assistant_turns
    )
    user_floor_seconds = sum(
        _overlap_seconds(crop_start_seconds, crop_end, start, end) for start, end in user_floor
    )
    return (
        not has_assistant_activity
        and user_floor_seconds >= crop_duration_seconds * 0.5
        and not _contains_event(
            crop_start_seconds,
            crop_duration_seconds,
            event_times,
            speculative_horizon_seconds,
        )
    )


def _intervals_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> bool:
    return first_start < second_end and second_start < first_end


def _overlap_seconds(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _assistant_probability_at(
    time_seconds: float,
    turns: tuple[ResolvedAssistantTurn, ...],
) -> float:
    probability = 0.0
    for turn in turns:
        if not turn.start_seconds <= time_seconds < turn.end_seconds:
            continue
        ramp = min(
            1.0,
            (time_seconds - turn.start_seconds) / max(turn.transition_seconds, FRAME_SECONDS),
            (turn.end_seconds - time_seconds) / max(turn.transition_seconds, FRAME_SECONDS),
        )
        turn_probability = turn.speaking_probability * max(0.0, ramp)
        for probability_dip in turn.probability_dips:
            dip_start = turn.start_seconds + probability_dip.start_offset_seconds
            dip_end = turn.start_seconds + probability_dip.end_offset_seconds
            if dip_start <= time_seconds < dip_end:
                turn_probability = min(turn_probability, probability_dip.probability)
        probability = max(probability, turn_probability)
    return probability


def _speculative_eot_target(
    time_seconds: float,
    horizon_seconds: float,
    completions: tuple[float, ...],
    duration_seconds: float,
) -> float:
    if time_seconds < 0.0 or time_seconds + horizon_seconds > duration_seconds:
        return MASKED_TARGET
    return float(
        any(
            time_seconds < completion <= time_seconds + horizon_seconds
            for completion in completions
        )
    )


def _sparse_targets(
    frame_times: tuple[float, ...],
    positive_events: tuple[float, ...],
    negative_events: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        1.0
        if _frame_contains(time_seconds, positive_events)
        else 0.0
        if _frame_contains(time_seconds, negative_events)
        else MASKED_TARGET
        for time_seconds in frame_times
    )


def _interval_targets(
    frame_times: tuple[float, ...],
    positive_intervals: tuple[tuple[float, float], ...],
    negative_events: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        1.0
        if _active_at(time_seconds, positive_intervals)
        else 0.0
        if _frame_contains(time_seconds, negative_events)
        else MASKED_TARGET
        for time_seconds in frame_times
    )


def _frame_contains(frame_center_seconds: float, event_times: tuple[float, ...]) -> bool:
    frame_start = frame_center_seconds - FRAME_SECONDS / 2.0
    frame_end = frame_center_seconds + FRAME_SECONDS / 2.0
    return any(frame_start <= event_time < frame_end for event_time in event_times)


def _active_at(time_seconds: float, intervals: tuple[tuple[float, float], ...]) -> float:
    return float(any(start <= time_seconds < end for start, end in intervals))


def _resample(samples: np.ndarray, source_rate_hz: int, output_rate_hz: int) -> np.ndarray:
    if source_rate_hz == output_rate_hz:
        return samples
    output_count = round(samples.size * output_rate_hz / source_rate_hz)
    source_positions = np.arange(samples.size, dtype=np.float64) / source_rate_hz
    output_positions = np.arange(output_count, dtype=np.float64) / output_rate_hz
    return np.interp(output_positions, source_positions, samples).astype(np.float32)


def _read_pcm16_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as audio_file:
        if audio_file.getsampwidth() != 2:
            raise ValueError(f"Rendered user clip must use 16-bit PCM: {path}")
        channel_count = audio_file.getnchannels()
        sample_rate_hz = audio_file.getframerate()
        frames = audio_file.readframes(audio_file.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channel_count > 1:
        samples = samples.reshape(-1, channel_count).mean(axis=1)
    return samples, sample_rate_hz


def _write_pcm16_wave(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    encoded = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate_hz)
        audio_file.writeframes(encoded.tobytes())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
