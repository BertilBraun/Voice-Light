from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from app.local.synthetic_generation.conversation_prompts import EnglishConversationPromptSet
from app.local.synthetic_generation.conversation_tts import (
    ConversationTtsManifest,
    RenderedConversationUserUnit,
    _write_manifest_atomically,
)
from app.local.synthetic_generation.conversation_voice_references import file_sha256


def merge_conversation_tts_shards(
    prompt_set_path: Path,
    reference_manifest_path: Path,
    shard_manifest_paths: tuple[Path, ...],
    output_directory: Path,
) -> ConversationTtsManifest:
    if not shard_manifest_paths:
        raise ValueError("At least one conversation TTS shard is required.")
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        prompt_set_path.read_text(encoding="utf-8")
    )
    manifests = tuple(_load_manifest(path) for path in shard_manifest_paths)
    first = manifests[0]
    _validate_manifests(
        manifests,
        prompt_set,
        prompt_set_path,
        reference_manifest_path,
    )
    units_by_key: dict[str, tuple[RenderedConversationUserUnit, Path]] = {}
    for manifest, manifest_path in zip(manifests, shard_manifest_paths, strict=True):
        for unit in manifest.rendered_units:
            key = _unit_key(unit)
            if key in units_by_key:
                raise ValueError(f"Conversation TTS shards repeat user unit {key}.")
            units_by_key[key] = (unit, manifest_path.parent)
    expected_keys = tuple(
        f"{plan.plan_id}:{prompt.unit_id}"
        for plan in prompt_set.plans
        for prompt in sorted(plan.user_prompts, key=lambda item: item.sequence_index)
    )
    missing = [key for key in expected_keys if key not in units_by_key]
    unexpected = sorted(set(units_by_key) - set(expected_keys))
    if missing or unexpected:
        raise ValueError(
            f"Conversation TTS shards do not cover the prompt set; missing={missing}, "
            f"unexpected={unexpected}."
        )
    audio_directory = output_directory / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)
    merged_units = tuple(_copy_unit(units_by_key[key], audio_directory) for key in expected_keys)
    clone_prompts_by_plan = {unit.plan_id: unit.clone_prompt for unit in merged_units}
    manifest = ConversationTtsManifest(
        prompt_set_id=prompt_set.set_id,
        prompt_set_sha256=first.prompt_set_sha256,
        reference_manifest_sha256=first.reference_manifest_sha256,
        backend=first.backend,
        detection=first.detection,
        clone_prompts=tuple(clone_prompts_by_plan[plan.plan_id] for plan in prompt_set.plans),
        rendered_units=merged_units,
    )
    records_path = output_directory / "rendered-units.jsonl"
    records_path.write_text(
        "".join(f"{unit.model_dump_json()}\n" for unit in merged_units),
        encoding="utf-8",
    )
    _write_manifest_atomically(output_directory / "render.json", manifest)
    return manifest


def _load_manifest(path: Path) -> ConversationTtsManifest:
    return ConversationTtsManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _validate_manifests(
    manifests: tuple[ConversationTtsManifest, ...],
    prompt_set: EnglishConversationPromptSet,
    prompt_set_path: Path,
    reference_manifest_path: Path,
) -> None:
    expected_prompt_hash = file_sha256(prompt_set_path)
    expected_reference_hash = file_sha256(reference_manifest_path)
    first = manifests[0]
    for manifest in manifests:
        if manifest.prompt_set_id != prompt_set.set_id:
            raise ValueError("Conversation TTS shard uses a different prompt set ID.")
        if manifest.prompt_set_sha256 != expected_prompt_hash:
            raise ValueError("Conversation TTS shard uses different prompt content.")
        if manifest.reference_manifest_sha256 != expected_reference_hash:
            raise ValueError("Conversation TTS shard uses different voice references.")
        if manifest.backend != first.backend or manifest.detection != first.detection:
            raise ValueError("Conversation TTS shards use different rendering configurations.")


def _copy_unit(
    source: tuple[RenderedConversationUserUnit, Path],
    audio_directory: Path,
) -> RenderedConversationUserUnit:
    unit, shard_directory = source
    source_path = shard_directory / unit.clip.audio_path
    if not source_path.exists() or file_sha256(source_path) != unit.clip.audio_sha256:
        raise ValueError(f"Conversation TTS shard audio is missing or changed: {source_path}.")
    destination_path = audio_directory / source_path.name
    if destination_path.exists():
        if file_sha256(destination_path) != unit.clip.audio_sha256:
            raise ValueError(f"Merged conversation audio conflicts at {destination_path}.")
    else:
        shutil.copy2(source_path, destination_path)
    relative_path = Path("audio") / destination_path.name
    return unit.model_copy(
        update={"clip": unit.clip.model_copy(update={"audio_path": relative_path})}
    )


def _unit_key(unit: RenderedConversationUserUnit) -> str:
    return f"{unit.plan_id}:{unit.prompt.unit_id}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge isolated conversation TTS shards.")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--shard-manifest", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    manifest = merge_conversation_tts_shards(
        prompt_set_path=parsed.prompts,
        reference_manifest_path=parsed.references,
        shard_manifest_paths=tuple(parsed.shard_manifest),
        output_directory=parsed.output,
    )
    print(manifest.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
