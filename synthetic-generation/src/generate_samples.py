from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from src.asr_placement import (
    WhisperTimestampTranscriber,
    measure_internal_pause_from_words,
    place_backchannel_from_words,
    place_interruption_from_words,
)
from src.audio_utils import audio_duration_seconds, audio_sample_rate, trim_to_speech
from src.backends import TTSBackend, TTSGenerationRequest, create_backend
from src.backends.base import TTSVariantResult
from src.interaction_cases import DEFAULT_CASES
from src.models import (
    BackchannelPlacementDefinition,
    BackendConfig,
    CaseManifest,
    CompletionPlacementDefinition,
    CompletionPlacementOutput,
    ExperimentManifest,
    GenerationBackendManifest,
    InteractionCase,
    InteractionCaseFile,
    InternalPausePlacementDefinition,
    InterruptionPlacementDefinition,
    PreparedRequest,
    SpeakerManifest,
    SpeakerPrompt,
    VariantManifest,
)

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


@dataclass(frozen=True)
class PlacementTarget:
    speaker_a_variant_id: str
    target_seconds: float
    placement_kind: str


def main() -> None:
    load_env_file(Path(".env"))
    arguments = parse_arguments()
    backend_config = load_backend_config(arguments)
    case_file = load_case_file(arguments.cases)
    experiment_id = arguments.experiment or case_file.experiment_id or "examples_v1"
    output_dir = Path("outputs") / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = create_backend(
        backend_config.backend,
        backend_config.model,
        backend_config.options,
    )
    try:
        backend.load()
        transcriber = build_transcriber(arguments)
        manifest = generate_experiment(
            backend=backend,
            backend_config=backend_config,
            case_file=case_file,
            experiment_id=experiment_id,
            output_dir=output_dir,
            variants=arguments.variants,
            seed=arguments.seed,
            transcriber=transcriber,
        )
    finally:
        backend.close()

    manifest_path = output_dir / "experiment.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote experiment manifest: {manifest_path}")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped_line = line.strip()
        if stripped_line == "" or stripped_line.startswith("#"):
            continue
        key, separator, raw_value = stripped_line.partition("=")
        if separator == "":
            continue
        normalized_key = key.strip()
        if normalized_key == "" or normalized_key in os.environ:
            continue
        os.environ[normalized_key] = normalize_env_value(raw_value.strip())


def normalize_env_value(raw_value: str) -> str:
    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {"'", '"'}:
        return raw_value[1:-1]
    return raw_value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic conversation TTS samples.")
    parser.add_argument("--cases", type=Path, default=None, help="JSON case definition file.")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment output id.")
    parser.add_argument("--variants", type=int, default=3, help="Variants per speaker.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed.")
    parser.add_argument("--backend", type=str, default="qwen-voice-design", help="TTS backend id.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Backend model id.")
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        help="Backend option as key=value. May be repeated.",
    )
    parser.add_argument("--backend-config", type=Path, default=None, help="Backend config JSON.")
    parser.add_argument(
        "--auto-place-backchannels",
        action="store_true",
        help="Deprecated alias for --auto-place.",
    )
    parser.add_argument(
        "--auto-place",
        action="store_true",
        help="Use ASR timestamps to compute suggested backchannel/interruption offsets.",
    )
    parser.add_argument("--asr-model", type=str, default="small", help="faster-whisper model size.")
    parser.add_argument("--asr-device", type=str, default="cpu", help="faster-whisper device.")
    parser.add_argument(
        "--asr-compute-type",
        type=str,
        default="int8",
        help="faster-whisper compute type.",
    )
    arguments = parser.parse_args()
    if arguments.variants < 1:
        raise ValueError("--variants must be at least 1")
    return arguments


def build_transcriber(arguments: argparse.Namespace) -> WhisperTimestampTranscriber | None:
    if not (arguments.auto_place or arguments.auto_place_backchannels):
        return None
    return WhisperTimestampTranscriber(
        model_size=arguments.asr_model,
        device=arguments.asr_device,
        compute_type=arguments.asr_compute_type,
    )


def load_backend_config(arguments: argparse.Namespace) -> BackendConfig:
    if arguments.backend_config is not None:
        try:
            config_data = json.loads(arguments.backend_config.read_text(encoding="utf-8"))
            file_config = BackendConfig.model_validate(config_data)
        except ValidationError as error:
            raise ValueError(str(error)) from error
        options = dict(file_config.options)
        options.update(parse_backend_options(arguments.backend_option))
        return BackendConfig(
            backend=arguments.backend or file_config.backend,
            model=arguments.model or file_config.model,
            options=options,
        )
    return BackendConfig(
        backend=arguments.backend,
        model=arguments.model,
        options=parse_backend_options(arguments.backend_option),
    )


def parse_backend_options(option_values: list[str]) -> dict[str, object]:
    options: dict[str, object] = {}
    for option_value in option_values:
        key, separator, raw_value = option_value.partition("=")
        if separator == "" or key == "":
            raise ValueError(f"Backend option must use key=value format: {option_value}")
        options[key] = parse_option_value(raw_value)
    return options


def parse_option_value(raw_value: str) -> object:
    lowered_value = raw_value.lower()
    if lowered_value == "true":
        return True
    if lowered_value == "false":
        return False
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def load_case_file(cases_path: Path | None) -> InteractionCaseFile:
    if cases_path is None:
        return DEFAULT_CASES
    try:
        case_data = json.loads(cases_path.read_text(encoding="utf-8-sig"))
        return InteractionCaseFile.model_validate(case_data)
    except ValidationError as error:
        raise ValueError(f"Case definition validation failed:\n{error}") from error


def generate_experiment(
    backend: TTSBackend,
    backend_config: BackendConfig,
    case_file: InteractionCaseFile,
    experiment_id: str,
    output_dir: Path,
    variants: int,
    seed: int | None,
    transcriber: WhisperTimestampTranscriber | None = None,
) -> ExperimentManifest:
    case_manifests: list[CaseManifest] = []
    for case_index, interaction_case in enumerate(case_file.cases):
        print(
            f"Generating case {case_index + 1}/{len(case_file.cases)}: {interaction_case.case_id}"
        )
        case_dir = output_dir / "cases" / interaction_case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case.json").write_text(
            interaction_case.model_dump_json(indent=2),
            encoding="utf-8",
        )
        speaker_a_manifest = generate_speaker(
            backend=backend,
            interaction_case=interaction_case,
            speaker_prompt=interaction_case.speaker_a,
            speaker_name="speaker_a",
            speaker_index=0,
            case_index=case_index,
            case_dir=case_dir,
            experiment_dir=output_dir,
            variants=variants,
            seed=seed,
        )
        if interaction_case.placement is not None:
            speaker_a_manifest = add_placement_outputs(
                speaker_manifest=speaker_a_manifest,
                placement_definition=interaction_case.placement,
                transcriber=transcriber,
                experiment_dir=output_dir,
                case_index=case_index,
                seed=seed,
            )
        speaker_b_manifest = generate_speaker(
            backend=backend,
            interaction_case=interaction_case,
            speaker_prompt=interaction_case.speaker_b,
            speaker_name="speaker_b",
            speaker_index=1,
            case_index=case_index,
            case_dir=case_dir,
            experiment_dir=output_dir,
            variants=variants,
            seed=seed,
        )
        if (
            interaction_case.placement is not None
            and interaction_case.placement.type != "internal_pause"
        ):
            speaker_b_manifest = add_speaker_b_target_offsets(
                speaker_a_manifest=speaker_a_manifest,
                speaker_b_manifest=speaker_b_manifest,
            )
        case_manifests.append(
            CaseManifest(
                case_id=interaction_case.case_id,
                title=interaction_case.title,
                description=interaction_case.description,
                alignment_notes=interaction_case.alignment_notes,
                tags=interaction_case.tags,
                default_b_offset_seconds=interaction_case.default_b_offset_seconds,
                placement=interaction_case.placement,
                speaker_a=speaker_a_manifest,
                speaker_b=speaker_b_manifest,
            )
        )
    return ExperimentManifest(
        experiment_id=experiment_id,
        description=case_file.description,
        generation_backend=GenerationBackendManifest(
            backend_id=backend.backend_id,
            model_id=backend.model_id,
            options=backend_config.options,
        ),
        cases=case_manifests,
    )


def generate_speaker(
    backend: TTSBackend,
    interaction_case: InteractionCase,
    speaker_prompt: SpeakerPrompt,
    speaker_name: str,
    speaker_index: int,
    case_index: int,
    case_dir: Path,
    experiment_dir: Path,
    variants: int,
    seed: int | None,
) -> SpeakerManifest:
    speaker_dir = case_dir / speaker_name
    request_seed = derive_request_seed(seed, case_index, speaker_index)
    result = backend.generate(
        TTSGenerationRequest(
            text=speaker_prompt.text,
            instruction=speaker_prompt.instruction,
            variants=variants,
            output_dir=speaker_dir,
            language="English",
            seed=request_seed,
            filename_prefix="variant",
            metadata={"case_id": interaction_case.case_id, "speaker": speaker_name},
        )
    )
    variant_manifests = [
        build_variant_manifest(
            variant_result=variant_result,
            backend_id=result.backend_id,
            model_id=result.model_id,
            speaker_prompt=speaker_prompt,
            experiment_dir=experiment_dir,
        )
        for variant_result in result.variants
    ]
    return SpeakerManifest(
        text=speaker_prompt.text,
        instruction=speaker_prompt.instruction,
        variants=variant_manifests,
    )


def add_placement_outputs(
    speaker_manifest: SpeakerManifest,
    placement_definition: (
        BackchannelPlacementDefinition
        | InterruptionPlacementDefinition
        | CompletionPlacementDefinition
        | InternalPausePlacementDefinition
    ),
    transcriber: WhisperTimestampTranscriber | None,
    experiment_dir: Path,
    case_index: int,
    seed: int | None,
) -> SpeakerManifest:
    match placement_definition:
        case BackchannelPlacementDefinition():
            require_transcriber(transcriber, placement_definition.type)
            return add_backchannel_placements(
                speaker_manifest=speaker_manifest,
                placement_definition=placement_definition,
                transcriber=transcriber,
                experiment_dir=experiment_dir,
            )
        case InterruptionPlacementDefinition():
            require_transcriber(transcriber, placement_definition.type)
            return add_interruption_placements(
                speaker_manifest=speaker_manifest,
                placement_definition=placement_definition,
                transcriber=transcriber,
                experiment_dir=experiment_dir,
            )
        case CompletionPlacementDefinition():
            return add_completion_placements(
                speaker_manifest=speaker_manifest,
                placement_definition=placement_definition,
                case_index=case_index,
                seed=seed,
            )
        case InternalPausePlacementDefinition():
            require_transcriber(transcriber, placement_definition.type)
            return add_internal_pause_measurements(
                speaker_manifest=speaker_manifest,
                placement_definition=placement_definition,
                transcriber=transcriber,
                experiment_dir=experiment_dir,
            )


def require_transcriber(
    transcriber: WhisperTimestampTranscriber | None,
    placement_type: str,
) -> None:
    if transcriber is None:
        raise ValueError(
            f"Placement type '{placement_type}' requires --auto-place so ASR timestamps "
            "are available."
        )


def add_backchannel_placements(
    speaker_manifest: SpeakerManifest,
    placement_definition: BackchannelPlacementDefinition,
    transcriber: WhisperTimestampTranscriber,
    experiment_dir: Path,
) -> SpeakerManifest:
    updated_variants: list[VariantManifest] = []
    for variant in speaker_manifest.variants:
        if variant.status != "success" or variant.audio_url is None:
            updated_variants.append(variant)
            continue
        audio_path = experiment_dir / Path(variant.audio_url)
        try:
            words = transcriber.transcribe_words(audio_path)
            suggested_offset_seconds, output = place_backchannel_from_words(
                words,
                placement_definition,
            )
            updated_variants.append(
                variant.model_copy(
                    update={
                        "suggested_b_offset_seconds": suggested_offset_seconds,
                        "placement_output": output,
                    }
                )
            )
        except Exception as error:
            backend_metadata = dict(variant.backend_metadata)
            backend_metadata["placement_error"] = str(error)
            updated_variants.append(
                variant.model_copy(update={"backend_metadata": backend_metadata})
            )
    return speaker_manifest.model_copy(update={"variants": updated_variants})


def add_interruption_placements(
    speaker_manifest: SpeakerManifest,
    placement_definition: InterruptionPlacementDefinition,
    transcriber: WhisperTimestampTranscriber,
    experiment_dir: Path,
) -> SpeakerManifest:
    updated_variants: list[VariantManifest] = []
    for variant in speaker_manifest.variants:
        if variant.status != "success" or variant.audio_url is None:
            updated_variants.append(variant)
            continue
        audio_path = experiment_dir / Path(variant.audio_url)
        try:
            words = transcriber.transcribe_words(audio_path)
            suggested_offset_seconds, output = place_interruption_from_words(
                words,
                placement_definition,
            )
            updated_variants.append(
                variant.model_copy(
                    update={
                        "suggested_b_offset_seconds": suggested_offset_seconds,
                        "placement_output": output,
                    }
                )
            )
        except Exception as error:
            backend_metadata = dict(variant.backend_metadata)
            backend_metadata["placement_error"] = str(error)
            updated_variants.append(
                variant.model_copy(update={"backend_metadata": backend_metadata})
            )
    return speaker_manifest.model_copy(update={"variants": updated_variants})


def add_completion_placements(
    speaker_manifest: SpeakerManifest,
    placement_definition: CompletionPlacementDefinition,
    case_index: int,
    seed: int | None,
) -> SpeakerManifest:
    min_delay_ms, max_delay_ms = completion_delay_range_ms(placement_definition)
    updated_variants: list[VariantManifest] = []
    for variant_index, variant in enumerate(speaker_manifest.variants):
        if variant.status != "success" or variant.duration_seconds is None:
            updated_variants.append(variant)
            continue
        sampled_delay_ms = sample_completion_delay_ms(
            min_delay_ms=min_delay_ms,
            max_delay_ms=max_delay_ms,
            seed=seed,
            case_index=case_index,
            variant_index=variant_index,
        )
        speaker_b_start_seconds = variant.duration_seconds + (sampled_delay_ms / 1000.0)
        output = CompletionPlacementOutput(
            pause=placement_definition.pause,
            min_delay_ms=min_delay_ms,
            max_delay_ms=max_delay_ms,
            sampled_delay_ms=sampled_delay_ms,
            speaker_a_end_seconds=variant.duration_seconds,
            speaker_b_start_seconds=speaker_b_start_seconds,
        )
        updated_variants.append(
            variant.model_copy(
                update={
                    "suggested_b_offset_seconds": speaker_b_start_seconds,
                    "placement_output": output,
                }
            )
        )
    return speaker_manifest.model_copy(update={"variants": updated_variants})


def add_internal_pause_measurements(
    speaker_manifest: SpeakerManifest,
    placement_definition: InternalPausePlacementDefinition,
    transcriber: WhisperTimestampTranscriber,
    experiment_dir: Path,
) -> SpeakerManifest:
    updated_variants: list[VariantManifest] = []
    for variant in speaker_manifest.variants:
        if variant.status != "success" or variant.audio_url is None:
            updated_variants.append(variant)
            continue
        audio_path = experiment_dir / Path(variant.audio_url)
        try:
            words = transcriber.transcribe_words(audio_path)
            output = measure_internal_pause_from_words(words, placement_definition)
            updated_variants.append(variant.model_copy(update={"placement_output": output}))
        except Exception as error:
            backend_metadata = dict(variant.backend_metadata)
            backend_metadata["placement_error"] = str(error)
            updated_variants.append(
                variant.model_copy(update={"backend_metadata": backend_metadata})
            )
    return speaker_manifest.model_copy(update={"variants": updated_variants})


def add_speaker_b_target_offsets(
    speaker_a_manifest: SpeakerManifest,
    speaker_b_manifest: SpeakerManifest,
) -> SpeakerManifest:
    targets = [
        PlacementTarget(
            speaker_a_variant_id=variant.id,
            target_seconds=variant.suggested_b_offset_seconds,
            placement_kind=placement_kind(variant),
        )
        for variant in speaker_a_manifest.variants
        if variant.status == "success" and variant.suggested_b_offset_seconds is not None
    ]
    if len(targets) == 0:
        return speaker_b_manifest
    updated_variants: list[VariantManifest] = []
    for index, variant in enumerate(speaker_b_manifest.variants):
        if variant.status != "success" or variant.audio_onset_seconds is None:
            updated_variants.append(variant)
            continue
        target = targets[min(index, len(targets) - 1)]
        suggested_file_start_seconds = target.target_seconds - variant.audio_onset_seconds
        placement_alignment_metadata = {
            "paired_speaker_a_variant_id": target.speaker_a_variant_id,
            "target_speaker_b_onset_seconds": target.target_seconds,
            "placement_kind": target.placement_kind,
            "speaker_b_audio_onset_seconds": variant.audio_onset_seconds,
            "speaker_b_audio_offset_seconds": variant.audio_offset_seconds,
            "suggested_file_start_seconds": suggested_file_start_seconds,
        }
        backend_metadata = dict(variant.backend_metadata)
        backend_metadata["speaker_b_onset_alignment"] = placement_alignment_metadata
        if target.placement_kind == "backchannel":
            backend_metadata["backchannel_onset_alignment"] = placement_alignment_metadata
        if target.placement_kind == "interruption":
            backend_metadata["interruption_onset_alignment"] = placement_alignment_metadata
        if target.placement_kind == "completion":
            backend_metadata["completion_onset_alignment"] = placement_alignment_metadata
        updated_variants.append(
            variant.model_copy(
                update={
                    "suggested_b_offset_seconds": suggested_file_start_seconds,
                    "backend_metadata": backend_metadata,
                }
            )
        )
    return speaker_b_manifest.model_copy(update={"variants": updated_variants})


def placement_kind(variant: VariantManifest) -> str:
    if variant.placement_output is None:
        return "unknown"
    return variant.placement_output.type


def completion_delay_range_ms(
    placement_definition: CompletionPlacementDefinition,
) -> tuple[int, int]:
    if (
        placement_definition.min_delay_ms is not None
        and placement_definition.max_delay_ms is not None
    ):
        return placement_definition.min_delay_ms, placement_definition.max_delay_ms
    match placement_definition.pause:
        case "short":
            return 200, 500
        case "medium":
            return 500, 1000
        case "long":
            return 1000, 2000


def sample_completion_delay_ms(
    min_delay_ms: int,
    max_delay_ms: int,
    seed: int | None,
    case_index: int,
    variant_index: int,
) -> int:
    if min_delay_ms > max_delay_ms:
        raise ValueError("completion placement min_delay_ms must be <= max_delay_ms.")
    random_seed = (seed or 0) + (case_index * 1000) + variant_index
    sampler = random.Random(random_seed)
    return sampler.randint(min_delay_ms, max_delay_ms)


def derive_request_seed(seed: int | None, case_index: int, speaker_index: int) -> int | None:
    if seed is None:
        return None
    return seed + (case_index * 1000) + (speaker_index * 100)


def build_variant_manifest(
    variant_result: TTSVariantResult,
    backend_id: str,
    model_id: str,
    speaker_prompt: SpeakerPrompt,
    experiment_dir: Path,
) -> VariantManifest:
    audio_url = None
    raw_audio_url = None
    filename = f"{variant_result.variant_id}.wav"
    duration_seconds = variant_result.duration_seconds
    sample_rate = variant_result.sample_rate
    audio_onset_seconds = None
    audio_offset_seconds = None
    original_duration_seconds = None
    trimmed_leading_seconds = None
    trimmed_trailing_seconds = None
    if variant_result.audio_path is not None:
        filename = variant_result.audio_path.name
        audio_url = variant_result.audio_path.relative_to(experiment_dir).as_posix()
    if variant_result.audio_path is not None and variant_result.status == "success":
        raw_audio_path = raw_cache_path(variant_result.audio_path)
        raw_audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(variant_result.audio_path, raw_audio_path)
        raw_audio_url = raw_audio_path.relative_to(experiment_dir).as_posix()
        speech_bounds = trim_to_speech(variant_result.audio_path)
        duration_seconds = audio_duration_seconds(variant_result.audio_path)
        sample_rate = audio_sample_rate(variant_result.audio_path)
        audio_onset_seconds = 0.0
        audio_offset_seconds = speech_bounds.duration_seconds
        original_duration_seconds = speech_bounds.original_duration_seconds
        trimmed_leading_seconds = speech_bounds.onset_seconds
        trimmed_trailing_seconds = (
            speech_bounds.original_duration_seconds - speech_bounds.offset_seconds
        )
    real_time_factor = None
    if duration_seconds and variant_result.generation_seconds is not None:
        real_time_factor = variant_result.generation_seconds / duration_seconds
    return VariantManifest(
        id=variant_result.variant_id,
        audio_url=audio_url,
        raw_audio_url=raw_audio_url,
        filename=filename,
        duration_seconds=duration_seconds,
        generation_seconds=variant_result.generation_seconds,
        real_time_factor=real_time_factor,
        seed=variant_result.seed,
        sample_rate=sample_rate,
        audio_onset_seconds=audio_onset_seconds,
        audio_offset_seconds=audio_offset_seconds,
        original_duration_seconds=original_duration_seconds,
        trimmed_leading_seconds=trimmed_leading_seconds,
        trimmed_trailing_seconds=trimmed_trailing_seconds,
        status="success" if variant_result.status == "success" else "failed",
        error=variant_result.error,
        backend_id=backend_id,
        model_id=model_id,
        request=speaker_prompt,
        prepared_request=PreparedRequest(
            text=variant_result.prepared_input.text,
            instruction=variant_result.prepared_input.instruction,
            parameters=variant_result.prepared_input.parameters,
        ),
        backend_metadata=variant_result.provider_metadata,
    )


def raw_cache_path(audio_path: Path) -> Path:
    return audio_path.parent / "raw" / audio_path.name


if __name__ == "__main__":
    main()
