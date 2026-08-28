from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.local.synthetic_generation.conversation_prompts import EnglishConversationPromptSet
from app.local.synthetic_generation.conversation_tts import ConversationTtsManifest
from app.local.synthetic_generation.conversation_voice_references import (
    ConversationVoiceReference,
    ConversationVoiceReferenceManifest,
)
from app.local.synthetic_generation.models import SyntheticModel

PublicationRunId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$")]


class PublishedFileRole(StrEnum):
    PROMPT_SET = "prompt_set"
    REFERENCE_MANIFEST = "reference_manifest"
    REFERENCE_AUDIO = "reference_audio"
    UNIT_MANIFEST = "unit_manifest"
    UNIT_AUDIO = "unit_audio"
    QUALITY_LEDGER = "quality_ledger"


class SyntheticPublicationRequest(SyntheticModel):
    run_id: PublicationRunId
    source_directory: Path
    staging_root: Path
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class PublishedFile(SyntheticModel):
    path: Path
    role: PublishedFileRole
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SyntheticRunPublicationManifest(SyntheticModel):
    schema_version: Literal["voice-light-synthetic-publication-v1"] = (
        "voice-light-synthetic-publication-v1"
    )
    generated_at: datetime
    run_id: PublicationRunId
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    prompt_set_id: str
    conversation_count: int = Field(gt=0)
    rendered_unit_count: int = Field(gt=0)
    files: tuple[PublishedFile, ...]


class SyntheticPublicationSummary(SyntheticModel):
    run_id: PublicationRunId
    destination: Path
    conversation_count: int = Field(gt=0)
    rendered_unit_count: int = Field(gt=0)
    file_count: int = Field(gt=0)
    total_size_bytes: int = Field(gt=0)


def summarize_synthetic_publication(
    manifest: SyntheticRunPublicationManifest,
    destination: Path,
) -> SyntheticPublicationSummary:
    return SyntheticPublicationSummary(
        run_id=manifest.run_id,
        destination=destination,
        conversation_count=manifest.conversation_count,
        rendered_unit_count=manifest.rendered_unit_count,
        file_count=len(manifest.files) + 1,
        total_size_bytes=sum(file.size_bytes for file in manifest.files)
        + (destination / "provenance.json").stat().st_size,
    )


def stage_synthetic_publication(
    request: SyntheticPublicationRequest,
) -> SyntheticRunPublicationManifest:
    destination = request.staging_root / "runs" / request.run_id
    if destination.exists():
        raise ValueError(f"Synthetic publication destination already exists: {destination}")
    _require_finished_source(request.source_directory)
    prompt_source = request.source_directory / "prompts.json"
    reference_manifest_source = request.source_directory / "references" / "voice-references.json"
    unit_manifest_source = request.source_directory / "cosyvoice" / "render.json"
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        prompt_source.read_text(encoding="utf-8")
    )
    reference_manifest = ConversationVoiceReferenceManifest.model_validate_json(
        reference_manifest_source.read_text(encoding="utf-8")
    )
    unit_manifest = ConversationTtsManifest.model_validate_json(
        unit_manifest_source.read_text(encoding="utf-8")
    )
    _validate_manifest_bindings(
        prompt_set,
        reference_manifest,
        unit_manifest,
        prompt_set_sha256=_file_sha256(prompt_source),
        reference_manifest_sha256=_file_sha256(reference_manifest_source),
    )
    destination.mkdir(parents=True)
    _copy_file(prompt_source, destination / "prompts.json")
    _stage_references(
        reference_manifest,
        reference_manifest_source,
        destination / "references",
    )
    portable_units = _stage_units(
        unit_manifest,
        unit_manifest_source,
        destination / "units",
    )
    (destination / "units" / "render.json").write_text(
        portable_units.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    for ledger_name in ("audit-prompts.json", "audit-complete.json"):
        _copy_file(
            request.source_directory / ledger_name,
            destination / ledger_name,
        )
    manifest = SyntheticRunPublicationManifest(
        generated_at=datetime.now(UTC),
        run_id=request.run_id,
        source_code_revision=request.source_code_revision,
        prompt_set_id=prompt_set.set_id,
        conversation_count=len(prompt_set.plans),
        rendered_unit_count=len(portable_units.rendered_units),
        files=_inventory(destination),
    )
    (destination / "provenance.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _require_finished_source(source_directory: Path) -> None:
    if not (source_directory / "CORPUS_COMPLETE").is_file():
        raise ValueError(f"Synthetic source corpus is not complete: {source_directory}")
    required = (
        source_directory / "prompts.json",
        source_directory / "references" / "voice-references.json",
        source_directory / "cosyvoice" / "render.json",
        source_directory / "audit-prompts.json",
        source_directory / "audit-complete.json",
    )
    missing = tuple(path for path in required if not path.is_file())
    if missing:
        raise ValueError(f"Synthetic source corpus is missing required files: {missing}")


def _validate_manifest_bindings(
    prompt_set: EnglishConversationPromptSet,
    references: ConversationVoiceReferenceManifest,
    units: ConversationTtsManifest,
    prompt_set_sha256: str,
    reference_manifest_sha256: str,
) -> None:
    if references.prompt_set_id != prompt_set.set_id or units.prompt_set_id != prompt_set.set_id:
        raise ValueError("Synthetic publication manifests have different prompt-set IDs.")
    if (
        references.prompt_set_sha256 != prompt_set_sha256
        or units.prompt_set_sha256 != prompt_set_sha256
    ):
        raise ValueError("Synthetic publication prompt-set hashes do not match.")
    if units.reference_manifest_sha256 != reference_manifest_sha256:
        raise ValueError("Synthetic publication reference-manifest hash does not match.")
    expected_plans = {plan.plan_id for plan in prompt_set.plans}
    reference_plans = {reference.plan_id for reference in references.references}
    unit_plans = {unit.plan_id for unit in units.rendered_units}
    if reference_plans != expected_plans or unit_plans != expected_plans:
        raise ValueError("Synthetic publication does not cover every conversation plan.")


def _stage_references(
    manifest: ConversationVoiceReferenceManifest,
    manifest_source: Path,
    destination: Path,
) -> None:
    destination.mkdir(parents=True)
    _copy_file(manifest_source, destination / "voice-references.json")
    for reference in manifest.references:
        source = _resolved_audio_path(manifest_source.parent, reference.audio_path)
        _validate_audio(source, reference.audio_sha256, "reference")
        _copy_file(source, destination / reference.audio_path)


def _stage_units(
    manifest: ConversationTtsManifest,
    manifest_source: Path,
    destination: Path,
) -> ConversationTtsManifest:
    destination.mkdir(parents=True)
    portable_units = []
    for unit in manifest.rendered_units:
        source = _resolved_audio_path(manifest_source.parent, unit.clip.audio_path)
        _validate_audio(source, unit.clip.audio_sha256, "speech unit")
        _copy_file(source, destination / unit.clip.audio_path)
        reference_path = Path("..") / "references" / "audio" / Path(unit.reference.audio_path).name
        portable_reference: ConversationVoiceReference = unit.reference.model_copy(
            update={"audio_path": reference_path}
        )
        portable_units.append(unit.model_copy(update={"reference": portable_reference}))
    return manifest.model_copy(update={"rendered_units": tuple(portable_units)})


def _resolved_audio_path(directory: Path, audio_path: Path) -> Path:
    return audio_path if audio_path.is_absolute() else directory / audio_path


def _validate_audio(path: Path, expected_sha256: str, description: str) -> None:
    if path.suffix.lower() != ".flac":
        raise ValueError(f"Published {description} must be FLAC: {path}")
    if not path.is_file():
        raise ValueError(f"Published {description} does not exist: {path}")
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"Published {description} hash changed: {path}")


def _copy_file(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"Synthetic publication file collision: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _inventory(destination: Path) -> tuple[PublishedFile, ...]:
    files = []
    for path in sorted(path for path in destination.rglob("*") if path.is_file()):
        relative_path = path.relative_to(destination)
        files.append(
            PublishedFile(
                path=relative_path,
                role=_file_role(relative_path),
                size_bytes=path.stat().st_size,
                sha256=_file_sha256(path),
            )
        )
    return tuple(files)


def _file_role(path: Path) -> PublishedFileRole:
    if path == Path("prompts.json"):
        return PublishedFileRole.PROMPT_SET
    if path == Path("references/voice-references.json"):
        return PublishedFileRole.REFERENCE_MANIFEST
    if path.parts[0] == "references":
        return PublishedFileRole.REFERENCE_AUDIO
    if path == Path("units/render.json"):
        return PublishedFileRole.UNIT_MANIFEST
    if path.parts[0] == "units":
        return PublishedFileRole.UNIT_AUDIO
    return PublishedFileRole.QUALITY_LEDGER


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
