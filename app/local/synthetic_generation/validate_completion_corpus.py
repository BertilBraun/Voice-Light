from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import Field

from app.local.synthetic_generation.completion_dataset import (
    CompletionBoundaryKind,
    GeneratedUtteranceAnnotation,
    PromptLanguage,
    TtsProvider,
)
from app.local.synthetic_generation.models import SyntheticModel


class CompletionValidationIssueCode(StrEnum):
    MISSING_AUDIO = "missing_audio"
    EMPTY_AUDIO = "empty_audio"
    NON_MONO_AUDIO = "non_mono_audio"
    SAMPLE_RATE_MISMATCH = "sample_rate_mismatch"
    DURATION_MISMATCH = "duration_mismatch"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    LOW_ENERGY = "low_energy"
    EXCESSIVE_CLIPPING = "excessive_clipping"
    NO_HOLD = "no_hold"


class CompletionValidationIssue(SyntheticModel):
    utterance_id: str
    code: CompletionValidationIssueCode
    detail: str


class CompletionProviderCount(SyntheticModel):
    provider: TtsProvider
    utterance_count: int = Field(gt=0)


class CompletionLanguageCount(SyntheticModel):
    language: PromptLanguage
    utterance_count: int = Field(gt=0)


class CompletionCorpusValidationReport(SyntheticModel):
    schema_version: str = "voice-light-synthetic-completion-validation-v1"
    corpus_directory: Path
    utterance_count: int = Field(ge=0)
    valid_utterance_count: int = Field(ge=0)
    generated_audio_seconds: float = Field(ge=0.0)
    hold_interval_count: int = Field(ge=0)
    provider_counts: tuple[CompletionProviderCount, ...]
    language_counts: tuple[CompletionLanguageCount, ...]
    issues: tuple[CompletionValidationIssue, ...]


def validate_completion_corpus(
    corpus_directory: Path,
    minimum_duration_seconds: float = 20.0,
    maximum_duration_seconds: float = 60.0,
    minimum_root_mean_square: float = 0.005,
    maximum_clipped_sample_fraction: float = 0.005,
) -> CompletionCorpusValidationReport:
    manifest_path = corpus_directory / "utterances.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"Completion manifest does not exist: {manifest_path}")
    annotations = tuple(
        GeneratedUtteranceAnnotation.model_validate_json(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    issues = []
    valid_utterance_count = 0
    for annotation in annotations:
        utterance_issues = _validate_utterance(
            corpus_directory=corpus_directory,
            annotation=annotation,
            minimum_duration_seconds=minimum_duration_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
            minimum_root_mean_square=minimum_root_mean_square,
            maximum_clipped_sample_fraction=maximum_clipped_sample_fraction,
        )
        issues.extend(utterance_issues)
        if not utterance_issues:
            valid_utterance_count += 1
    provider_counts = Counter(annotation.provenance.provider for annotation in annotations)
    language_counts = Counter(annotation.prompt.language for annotation in annotations)
    return CompletionCorpusValidationReport(
        corpus_directory=corpus_directory,
        utterance_count=len(annotations),
        valid_utterance_count=valid_utterance_count,
        generated_audio_seconds=sum(
            annotation.trimmed_duration_seconds for annotation in annotations
        ),
        hold_interval_count=sum(
            sum(boundary.kind is CompletionBoundaryKind.HOLD for boundary in annotation.boundaries)
            for annotation in annotations
        ),
        provider_counts=tuple(
            CompletionProviderCount(provider=provider, utterance_count=count)
            for provider, count in sorted(provider_counts.items())
        ),
        language_counts=tuple(
            CompletionLanguageCount(language=language, utterance_count=count)
            for language, count in sorted(language_counts.items())
        ),
        issues=tuple(issues),
    )


def _validate_utterance(
    corpus_directory: Path,
    annotation: GeneratedUtteranceAnnotation,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    minimum_root_mean_square: float,
    maximum_clipped_sample_fraction: float,
) -> tuple[CompletionValidationIssue, ...]:
    issues = []
    audio_path = corpus_directory / annotation.audio_path
    if not audio_path.is_file():
        return (
            _issue(
                annotation,
                CompletionValidationIssueCode.MISSING_AUDIO,
                f"Audio does not exist: {audio_path}",
            ),
        )
    decoded_audio, sample_rate_hz = sf.read(audio_path, dtype="float32", always_2d=False)
    decoded_audio = np.asarray(decoded_audio, dtype=np.float32)
    if decoded_audio.ndim != 1:
        return (
            _issue(
                annotation,
                CompletionValidationIssueCode.NON_MONO_AUDIO,
                f"Audio has {decoded_audio.shape[1]} channels.",
            ),
        )
    samples = decoded_audio
    if samples.size == 0:
        return (_issue(annotation, CompletionValidationIssueCode.EMPTY_AUDIO, "Audio is empty."),)
    if sample_rate_hz != annotation.sample_rate_hz:
        issues.append(
            _issue(
                annotation,
                CompletionValidationIssueCode.SAMPLE_RATE_MISMATCH,
                f"Manifest={annotation.sample_rate_hz}, WAV={sample_rate_hz}.",
            )
        )
    duration_seconds = samples.size / sample_rate_hz
    if not np.isclose(duration_seconds, annotation.trimmed_duration_seconds, atol=0.021):
        issues.append(
            _issue(
                annotation,
                CompletionValidationIssueCode.DURATION_MISMATCH,
                f"Manifest={annotation.trimmed_duration_seconds:.3f}, WAV={duration_seconds:.3f}.",
            )
        )
    if duration_seconds < minimum_duration_seconds:
        issues.append(
            _issue(
                annotation,
                CompletionValidationIssueCode.TOO_SHORT,
                f"Duration {duration_seconds:.3f}s is below {minimum_duration_seconds:.3f}s.",
            )
        )
    if duration_seconds > maximum_duration_seconds:
        issues.append(
            _issue(
                annotation,
                CompletionValidationIssueCode.TOO_LONG,
                f"Duration {duration_seconds:.3f}s exceeds {maximum_duration_seconds:.3f}s.",
            )
        )
    root_mean_square = float(np.sqrt(np.mean(np.square(samples))))
    if root_mean_square < minimum_root_mean_square:
        issues.append(
            _issue(
                annotation,
                CompletionValidationIssueCode.LOW_ENERGY,
                f"RMS {root_mean_square:.6f} is below {minimum_root_mean_square:.6f}.",
            )
        )
    clipped_fraction = float(np.mean(np.abs(samples) >= 0.999))
    if clipped_fraction > maximum_clipped_sample_fraction:
        issues.append(
            _issue(
                annotation,
                CompletionValidationIssueCode.EXCESSIVE_CLIPPING,
                f"Clipped fraction {clipped_fraction:.6f} exceeds "
                f"{maximum_clipped_sample_fraction:.6f}.",
            )
        )
    if not any(boundary.kind is CompletionBoundaryKind.HOLD for boundary in annotation.boundaries):
        issues.append(
            _issue(
                annotation,
                CompletionValidationIssueCode.NO_HOLD,
                "No internal silence of at least 500 ms was detected.",
            )
        )
    return tuple(issues)


def _issue(
    annotation: GeneratedUtteranceAnnotation,
    code: CompletionValidationIssueCode,
    detail: str,
) -> CompletionValidationIssue:
    return CompletionValidationIssue(
        utterance_id=annotation.utterance_id,
        code=code,
        detail=detail,
    )


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    report = validate_completion_corpus(parsed.corpus)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2), flush=True)
    if report.issues:
        raise SystemExit(1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a generated completion corpus.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
