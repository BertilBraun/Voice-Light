from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from app.local.db.models import TrackSide
from app.local.synthetic_generation.conversation_compiler import (
    AfterAssistantUserTiming,
    AfterUserAssistantTiming,
    AfterUserUserTiming,
    AssistantProbabilityDip,
    CompiledConversation,
    CompletionPlacement,
    ConversationCompilerConfig,
    ConversationCompositionPlan,
    DuringAssistantUserTiming,
    FixedAssistantTiming,
    FixedUserTiming,
    HoldPlacement,
    InterruptionFloorClaimPlacement,
    NonFloorFeedbackPlacement,
    RenderedUserClip,
    ResponseFloorClaimPlacement,
    TrainingCropPlan,
    VirtualAssistantTurn,
    compile_conversation,
    materialize_crop_audio,
)
from app.local.synthetic_generation.conversation_prompts import (
    AssistantTurnPrompt,
    CompletionUserPrompt,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    HoldUserPrompt,
    InterruptionFloorClaimUserPrompt,
    NonFloorFeedbackUserPrompt,
    ResponseFloorClaimUserPrompt,
    UserPrompt,
)
from app.local.synthetic_generation.conversation_tts import ConversationTtsManifest
from app.local.synthetic_generation.models import SyntheticModel
from app.local.training_corpus.export import (
    ExportManifest,
    ExportShard,
    ExportSplitSummary,
    MaterializedTrainingSample,
    write_training_shards,
)
from app.local.training_corpus.splits import (
    ConversationSplitCandidate,
    ConversationSplitPlan,
    TrainingCorpusSplit,
    assign_conversation_splits,
)
from app.local.training_samples.constants import FRAME_SECONDS, INPUT_DURATION_SECONDS

SYNTHETIC_SCHEMA_VERSION = "voice-light-synthetic-turn-taking-v2"
SYNTHETIC_LABEL_VERSION = "semantic-user-floor-v1"
SYNTHETIC_DATASET_NAMESPACE = UUID("20af1d75-157b-4f2c-a2f9-53c4c6f737c7")
MASKED_TARGET = -1.0


class CompiledConversationReference(SyntheticModel):
    conversation_id: str
    split: TrainingCorpusSplit
    source_duration_seconds: float = Field(gt=0.0)
    plan_path: str
    manifest_path: str
    crop_audio_paths: tuple[str, ...]


class CropSamplingSummary(SyntheticModel):
    total_crop_count: int = Field(gt=0)
    event_focused_count: int = Field(ge=0)
    assistant_only_count: int = Field(ge=0)
    user_only_count: int = Field(ge=0)
    event_light_count: int = Field(ge=0)
    padded_count: int = Field(ge=0)
    assistant_only_fraction: float = Field(ge=0.0, le=1.0)
    user_only_fraction: float = Field(ge=0.0, le=1.0)
    event_light_fraction: float = Field(ge=0.0, le=1.0)
    padded_fraction: float = Field(ge=0.0, le=1.0)
    control_quotas_satisfied: bool
    padding_limit_satisfied: bool

    @model_validator(mode="after")
    def validate_gate_results(self) -> CropSamplingSummary:
        expected_controls = (
            self.assistant_only_fraction >= 0.1
            and self.user_only_fraction >= 0.1
            and self.event_light_fraction >= 0.1
        )
        if self.control_quotas_satisfied != expected_controls:
            raise ValueError("Control quota status does not match reported crop fractions.")
        if self.padding_limit_satisfied != (self.padded_fraction <= 0.05):
            raise ValueError("Padding limit status does not match the reported crop fraction.")
        return self


class SyntheticConversationCorpusManifest(SyntheticModel):
    schema_version: Literal["voice-light-synthetic-conversation-corpus-v1"] = (
        "voice-light-synthetic-conversation-corpus-v1"
    )
    prompt_set_id: str
    prompt_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tts_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: UUID
    split_plan: ConversationSplitPlan
    conversations: tuple[CompiledConversationReference, ...]
    sampling_summary: CropSamplingSummary


def build_conversation_corpus(
    prompt_set_path: Path,
    tts_manifest_path: Path,
    output_directory: Path,
    split_seed: str,
    compiler_config: ConversationCompilerConfig,
) -> SyntheticConversationCorpusManifest:
    if not split_seed:
        raise ValueError("Synthetic split seed must not be empty.")
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        prompt_set_path.read_text(encoding="utf-8")
    )
    tts_manifest = ConversationTtsManifest.model_validate_json(
        tts_manifest_path.read_text(encoding="utf-8")
    )
    prompt_set_sha256 = _file_sha256(prompt_set_path)
    if tts_manifest.prompt_set_id != prompt_set.set_id:
        raise ValueError("TTS manifest belongs to a different prompt set.")
    if tts_manifest.prompt_set_sha256 != prompt_set_sha256:
        raise ValueError("TTS manifest prompt-set hash does not match the input plan.")
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset_id = uuid5(SYNTHETIC_DATASET_NAMESPACE, f"{prompt_set.set_id}:{prompt_set_sha256}")
    sample_id_by_plan = {plan.plan_id: uuid5(dataset_id, plan.plan_id) for plan in prompt_set.plans}
    split_plan = assign_conversation_splits(
        tuple(
            ConversationSplitCandidate(
                dataset_id=dataset_id, sample_id=sample_id_by_plan[plan.plan_id]
            )
            for plan in prompt_set.plans
        ),
        split_seed,
    )
    split_by_sample = {
        assignment.sample_id: assignment.split for assignment in split_plan.assignments
    }
    rendered_by_plan = _rendered_units_by_plan(prompt_set, tts_manifest, tts_manifest_path.parent)
    samples: list[MaterializedTrainingSample] = []
    compiled_crops: list[TrainingCropPlan] = []
    references: list[CompiledConversationReference] = []
    for prompt_plan in prompt_set.plans:
        rendered_clips = rendered_by_plan[prompt_plan.plan_id]
        composition_plan = composition_plan_from_rendered_prompt(prompt_plan, rendered_clips)
        conversation_directory = output_directory / "conversations" / prompt_plan.plan_id
        compiled = compile_conversation(
            composition_plan,
            rendered_clips,
            conversation_directory / "source.wav",
            compiler_config,
        )
        plan_path = conversation_directory / "composition.json"
        compiled_path = conversation_directory / "compiled.json"
        _write_json(plan_path, compiled.plan)
        crop_paths = tuple(
            _materialize_crop(
                output_directory,
                conversation_directory,
                compiled,
                crop,
            )
            for crop in compiled.crops
        )
        compiled_crops.extend(compiled.crops)
        _write_json(compiled_path, compiled)
        split = split_by_sample[sample_id_by_plan[prompt_plan.plan_id]]
        samples.extend(
            _training_sample(
                dataset_id=dataset_id,
                sample_id=sample_id_by_plan[prompt_plan.plan_id],
                dataset_name=prompt_set.set_id,
                plan=prompt_plan,
                crop=crop,
                crop_audio_path=crop_path.relative_to(output_directory),
                split=split,
            )
            for crop, crop_path in zip(compiled.crops, crop_paths, strict=True)
        )
        references.append(
            CompiledConversationReference(
                conversation_id=prompt_plan.plan_id,
                split=split,
                source_duration_seconds=compiled.plan.duration_seconds,
                plan_path=plan_path.relative_to(output_directory).as_posix(),
                manifest_path=compiled_path.relative_to(output_directory).as_posix(),
                crop_audio_paths=tuple(
                    path.relative_to(output_directory).as_posix() for path in crop_paths
                ),
            )
        )
    sampling_summary = summarize_crop_sampling(tuple(compiled_crops))
    validate_sampling_gates(sampling_summary)
    shards = write_training_shards(output_directory, samples)
    export_manifest = _export_manifest(
        prompt_set,
        split_plan,
        tuple(samples),
        shards,
        compiler_config,
        tuple(references),
    )
    (output_directory / "corpus.json").write_text(
        export_manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    manifest = SyntheticConversationCorpusManifest(
        prompt_set_id=prompt_set.set_id,
        prompt_set_sha256=prompt_set_sha256,
        tts_manifest_sha256=_file_sha256(tts_manifest_path),
        dataset_id=dataset_id,
        split_plan=split_plan,
        conversations=tuple(references),
        sampling_summary=sampling_summary,
    )
    (output_directory / "synthetic-corpus.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def composition_plan_from_rendered_prompt(
    prompt_plan: EnglishConversationPromptPlan,
    rendered_clips: tuple[RenderedUserClip, ...],
) -> ConversationCompositionPlan:
    clip_by_unit_id = {
        clip.clip_id.removeprefix(f"{prompt_plan.plan_id}_"): clip for clip in rendered_clips
    }
    expected_unit_ids = {prompt.unit_id for prompt in prompt_plan.user_prompts}
    if set(clip_by_unit_id) != expected_unit_ids:
        raise ValueError(f"Rendered clips do not exactly cover plan {prompt_plan.plan_id}.")
    assistant_turns = tuple(
        _assistant_turn(prompt_plan, turn, prompt_plan.user_prompts)
        for turn in prompt_plan.assistant_turns
    )
    user_events = tuple(
        _user_placement(prompt_plan, prompt, clip_by_unit_id[prompt.unit_id])
        for prompt in prompt_plan.user_prompts
    )
    return ConversationCompositionPlan(
        conversation_id=prompt_plan.plan_id,
        seed=prompt_plan.seed,
        duration_seconds=prompt_plan.target_duration_seconds,
        user_events=user_events,
        assistant_turns=assistant_turns,
    )


def _assistant_turn(
    plan: EnglishConversationPromptPlan,
    prompt: AssistantTurnPrompt,
    user_prompts: tuple[UserPrompt, ...],
) -> VirtualAssistantTurn:
    prior_users = tuple(
        sorted(
            (
                user_prompt
                for user_prompt in user_prompts
                if user_prompt.sequence_index < prompt.sequence_index and _owns_floor(user_prompt)
            ),
            key=lambda user_prompt: user_prompt.sequence_index,
        )
    )
    generator = random.Random(plan.seed + prompt.sequence_index * 7_919)
    if prior_users:
        timing = AfterUserAssistantTiming(
            user_event_id=prior_users[-1].unit_id,
            delay_seconds=generator.uniform(0.12, 0.65),
        )
    else:
        timing = FixedAssistantTiming(start_seconds=generator.uniform(0.1, 0.5))
    duration = _estimated_assistant_duration(prompt)
    dips: list[AssistantProbabilityDip] = []
    for item in user_prompts:
        match item:
            case NonFloorFeedbackUserPrompt(during_assistant_turn_id=referenced_turn_id) if (
                referenced_turn_id == prompt.turn_id and duration > 0.5
            ):
                event_fraction = _assistant_event_fraction(plan, item)
                dips.append(
                    AssistantProbabilityDip(
                        start_offset_seconds=max(0.08, duration * event_fraction - 0.12),
                        end_offset_seconds=min(duration - 0.08, duration * event_fraction + 0.3),
                        probability=generator.uniform(0.58, 0.78),
                    )
                )
            case _:
                pass
    return VirtualAssistantTurn(
        turn_id=prompt.turn_id,
        timing=timing,
        duration_seconds=duration,
        speaking_probability=generator.uniform(0.88, 1.0),
        transition_seconds=generator.uniform(0.08, 0.32),
        probability_dips=tuple(dips),
    )


def _user_placement(
    plan: EnglishConversationPromptPlan,
    prompt: UserPrompt,
    clip: RenderedUserClip,
) -> (
    CompletionPlacement
    | HoldPlacement
    | NonFloorFeedbackPlacement
    | ResponseFloorClaimPlacement
    | InterruptionFloorClaimPlacement
):
    prior_assistants = tuple(
        sorted(
            (turn for turn in plan.assistant_turns if turn.sequence_index < prompt.sequence_index),
            key=lambda turn: turn.sequence_index,
        )
    )
    prior_floor_users = tuple(
        sorted(
            (
                user_prompt
                for user_prompt in plan.user_prompts
                if user_prompt.sequence_index < prompt.sequence_index and _owns_floor(user_prompt)
            ),
            key=lambda user_prompt: user_prompt.sequence_index,
        )
    )
    generator = random.Random(plan.seed + prompt.sequence_index * 15_497)
    match prompt:
        case NonFloorFeedbackUserPrompt(during_assistant_turn_id=assistant_turn_id):
            return NonFloorFeedbackPlacement(
                event_id=prompt.unit_id,
                clip_id=clip.clip_id,
                timing=DuringAssistantUserTiming(
                    assistant_turn_id=assistant_turn_id,
                    position_fraction=_assistant_event_fraction(plan, prompt),
                ),
            )
        case InterruptionFloorClaimUserPrompt(
            during_assistant_turn_id=assistant_turn_id,
            assistant_yield_delay_seconds=yield_delay_seconds,
        ):
            return InterruptionFloorClaimPlacement(
                event_id=prompt.unit_id,
                clip_id=clip.clip_id,
                timing=DuringAssistantUserTiming(
                    assistant_turn_id=assistant_turn_id,
                    position_fraction=_assistant_event_fraction(plan, prompt),
                ),
                assistant_yield_delay_seconds=yield_delay_seconds,
            )
        case ResponseFloorClaimUserPrompt(
            after_assistant_turn_id=assistant_turn_id,
            response_latency_seconds=response_latency_seconds,
        ):
            return ResponseFloorClaimPlacement(
                event_id=prompt.unit_id,
                clip_id=clip.clip_id,
                timing=AfterAssistantUserTiming(
                    assistant_turn_id=assistant_turn_id,
                    delay_seconds=response_latency_seconds,
                ),
            )
        case CompletionUserPrompt():
            timing = _floor_owning_timing(
                prior_floor_users,
                prior_assistants,
                generator,
            )
            return CompletionPlacement(event_id=prompt.unit_id, clip_id=clip.clip_id, timing=timing)
        case HoldUserPrompt() if clip.continuation_silences:
            timing = _floor_owning_timing(
                prior_floor_users,
                prior_assistants,
                generator,
            )
            return HoldPlacement(event_id=prompt.unit_id, clip_id=clip.clip_id, timing=timing)
        case HoldUserPrompt():
            timing = _floor_owning_timing(
                prior_floor_users,
                prior_assistants,
                generator,
            )
            return CompletionPlacement(event_id=prompt.unit_id, clip_id=clip.clip_id, timing=timing)


def _floor_owning_timing(
    prior_users: tuple[UserPrompt, ...],
    prior_assistants: tuple[AssistantTurnPrompt, ...],
    generator: random.Random,
) -> FixedUserTiming | AfterAssistantUserTiming | AfterUserUserTiming:
    latest_user = prior_users[-1] if prior_users else None
    latest_assistant = prior_assistants[-1] if prior_assistants else None
    delay_seconds = generator.uniform(0.12, 1.2)
    if latest_user is not None and (
        latest_assistant is None or latest_user.sequence_index > latest_assistant.sequence_index
    ):
        return AfterUserUserTiming(
            user_event_id=latest_user.unit_id,
            delay_seconds=delay_seconds,
        )
    if latest_assistant is not None:
        return AfterAssistantUserTiming(
            assistant_turn_id=latest_assistant.turn_id,
            delay_seconds=delay_seconds,
        )
    return FixedUserTiming(start_seconds=generator.uniform(0.1, 0.7))


def _estimated_assistant_duration(prompt: AssistantTurnPrompt) -> float:
    word_duration = len(prompt.text.split()) * 60.0 / prompt.speaking_rate_words_per_minute
    punctuation_count = sum(prompt.text.count(mark) for mark in (",", ";", ":", ".", "?", "!"))
    punctuation_duration = prompt.punctuation_pause_seconds * max(1, punctuation_count)
    return word_duration + punctuation_duration


def _owns_floor(prompt: UserPrompt) -> bool:
    match prompt:
        case NonFloorFeedbackUserPrompt():
            return False
        case (
            CompletionUserPrompt()
            | HoldUserPrompt()
            | ResponseFloorClaimUserPrompt()
            | InterruptionFloorClaimUserPrompt()
        ):
            return True


def _assistant_event_fraction(
    plan: EnglishConversationPromptPlan,
    prompt: NonFloorFeedbackUserPrompt | InterruptionFloorClaimUserPrompt,
) -> float:
    generator = random.Random(plan.seed + prompt.sequence_index * 31_337)
    return generator.uniform(0.28, 0.72)


def _rendered_units_by_plan(
    prompt_set: EnglishConversationPromptSet,
    tts_manifest: ConversationTtsManifest,
    tts_directory: Path,
) -> dict[str, tuple[RenderedUserClip, ...]]:
    clips: dict[str, list[RenderedUserClip]] = {plan.plan_id: [] for plan in prompt_set.plans}
    for unit in tts_manifest.rendered_units:
        if unit.plan_id not in clips:
            raise ValueError(f"TTS manifest contains unknown plan {unit.plan_id}.")
        clips[unit.plan_id].append(
            unit.clip.model_copy(update={"audio_path": tts_directory / unit.clip.audio_path})
        )
    return {plan_id: tuple(plan_clips) for plan_id, plan_clips in clips.items()}


def _materialize_crop(
    output_directory: Path,
    conversation_directory: Path,
    compiled: CompiledConversation,
    crop: TrainingCropPlan,
) -> Path:
    path = conversation_directory / "crops" / f"{crop.variant_index:03d}.wav"
    materialize_crop_audio(compiled, crop, path)
    return path


def _training_sample(
    dataset_id: UUID,
    sample_id: UUID,
    dataset_name: str,
    plan: EnglishConversationPromptPlan,
    crop: TrainingCropPlan,
    crop_audio_path: Path,
    split: TrainingCorpusSplit,
) -> MaterializedTrainingSample:
    labels = crop.labels
    masked = (MASKED_TARGET,) * len(labels.p_user_floor_now)
    speculative_by_horizon = {
        round(track.horizon_seconds * 1000): track.probabilities for track in labels.speculative_eot
    }
    return MaterializedTrainingSample(
        schema_version=SYNTHETIC_SCHEMA_VERSION,
        training_label_version=SYNTHETIC_LABEL_VERSION,
        window_id=crop.crop_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        sample_id=sample_id,
        external_id=plan.plan_id,
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        split=split,
        user_audio_path=crop_audio_path.as_posix(),
        assistant_audio_path=None,
        start_seconds=0.0,
        end_seconds=INPUT_DURATION_SECONDS,
        quality_score=1.0,
        category=f"synthetic_conversation:{crop.sampling_stratum.value}",
        assistant_only_control=crop.assistant_only,
        user_only_control=crop.user_only,
        event_light_control=crop.event_light,
        padded=crop.padded,
        assistant_has_floor=labels.assistant_speaking_probability,
        assistant_speaking_probability=labels.assistant_speaking_probability,
        p_user_has_floor=labels.p_user_floor_now,
        p_user_floor_now=labels.p_user_floor_now,
        p_user_yield=tuple(
            MASKED_TARGET if value == MASKED_TARGET else 1.0 - value
            for value in labels.p_user_floor_now
        ),
        p_assistant_backchannel=masked,
        future_activity_0_200=masked,
        future_activity_200_500=masked,
        future_activity_500_1000=masked,
        future_activity_1000_1500=masked,
        turn_completion=labels.turn_completion,
        continuation_pause=labels.continuation_pause,
        non_floor_feedback=labels.non_floor_feedback,
        floor_take=labels.floor_take,
        speculative_eot_500=speculative_by_horizon.get(500),
        speculative_eot_1000=speculative_by_horizon.get(1000),
    )


def _export_manifest(
    prompt_set: EnglishConversationPromptSet,
    split_plan: ConversationSplitPlan,
    samples: tuple[MaterializedTrainingSample, ...],
    shards: tuple[ExportShard, ...],
    config: ConversationCompilerConfig,
    conversations: tuple[CompiledConversationReference, ...],
) -> ExportManifest:
    return ExportManifest(
        schema_version=SYNTHETIC_SCHEMA_VERSION,
        generated_at=datetime.now(UTC),
        metric_version="synthetic-semantic-v1",
        annotation_version="synthetic-conversation-plan-v1",
        region_analysis_version="not-applicable",
        training_label_version=SYNTHETIC_LABEL_VERSION,
        input_duration_seconds=config.crop_duration_seconds,
        frame_seconds=FRAME_SECONDS,
        review_set_name=prompt_set.set_id,
        split_plan=split_plan,
        recording_count=len(conversations),
        training_sample_count=len(samples),
        splits=tuple(
            ExportSplitSummary(
                split=split,
                recording_count=sum(conversation.split is split for conversation in conversations),
                training_sample_count=sum(sample.split is split for sample in samples),
                source_duration_seconds=sum(
                    conversation.source_duration_seconds
                    for conversation in conversations
                    if conversation.split is split
                ),
            )
            for split in TrainingCorpusSplit
        ),
        shards=shards,
    )


def summarize_crop_sampling(crops: tuple[TrainingCropPlan, ...]) -> CropSamplingSummary:
    if not crops:
        raise ValueError("Synthetic corpus requires at least one materialized crop.")
    total = len(crops)
    event_focused_count = sum(not crop.event_light for crop in crops)
    assistant_only_count = sum(crop.assistant_only for crop in crops)
    user_only_count = sum(crop.user_only for crop in crops)
    event_light_count = sum(crop.event_light for crop in crops)
    padded_count = sum(crop.padded for crop in crops)
    assistant_only_fraction = assistant_only_count / total
    user_only_fraction = user_only_count / total
    event_light_fraction = event_light_count / total
    padded_fraction = padded_count / total
    return CropSamplingSummary(
        total_crop_count=total,
        event_focused_count=event_focused_count,
        assistant_only_count=assistant_only_count,
        user_only_count=user_only_count,
        event_light_count=event_light_count,
        padded_count=padded_count,
        assistant_only_fraction=assistant_only_fraction,
        user_only_fraction=user_only_fraction,
        event_light_fraction=event_light_fraction,
        padded_fraction=padded_fraction,
        control_quotas_satisfied=(
            assistant_only_fraction >= 0.1
            and user_only_fraction >= 0.1
            and event_light_fraction >= 0.1
        ),
        padding_limit_satisfied=padded_fraction <= 0.05,
    )


def validate_sampling_gates(summary: CropSamplingSummary) -> None:
    if not summary.control_quotas_satisfied:
        raise ValueError(
            "Synthetic crop controls miss the 10% corpus quota: "
            f"assistant_only={summary.assistant_only_fraction:.3f}, "
            f"user_only={summary.user_only_fraction:.3f}, "
            f"event_light={summary.event_light_fraction:.3f}."
        )
    if not summary.padding_limit_satisfied:
        raise ValueError(
            f"Synthetic padded crop fraction exceeds 5%: padded={summary.padded_fraction:.3f}."
        )


def _write_json(path: Path, value: SyntheticModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile rendered user-only conversations into Voice-Light training shards."
    )
    parser.add_argument("--prompt-set", type=Path, required=True)
    parser.add_argument("--tts-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", default="synthetic-conversation-pilot-v1")
    parser.add_argument("--crop-variants", type=int, default=8)
    parser.add_argument("--assistant-only-fraction", type=float, default=0.1)
    parser.add_argument("--user-only-fraction", type=float, default=0.1)
    parser.add_argument("--event-light-fraction", type=float, default=0.1)
    parser.add_argument("--assistant-duration-variation", type=float, default=0.1)
    arguments = parser.parse_args()
    manifest = build_conversation_corpus(
        prompt_set_path=arguments.prompt_set,
        tts_manifest_path=arguments.tts_manifest,
        output_directory=arguments.output,
        split_seed=arguments.split_seed,
        compiler_config=ConversationCompilerConfig(
            crop_variant_count=arguments.crop_variants,
            assistant_only_fraction=arguments.assistant_only_fraction,
            user_only_fraction=arguments.user_only_fraction,
            event_light_fraction=arguments.event_light_fraction,
            assistant_duration_variation=arguments.assistant_duration_variation,
        ),
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
