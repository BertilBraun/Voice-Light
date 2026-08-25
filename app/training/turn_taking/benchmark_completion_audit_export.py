from __future__ import annotations

import csv
import html
import json
import time
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TextIO

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from app.local.training_corpus.export import MaterializedTrainingSample
from app.training.turn_taking.benchmark_adapters import AudioPathResolver
from app.training.turn_taking.benchmark_completion_audit import CompletionAuditManifest
from app.training.turn_taking.benchmark_corpus_quality_audit import (
    CorpusQualityAuditManifest,
)
from app.training.turn_taking.benchmark_models import BenchmarkModel, TurnCompletionInventory
from app.training.turn_taking.data import load_audio_window

AUDIT_SAMPLE_RATE_HZ = 16_000
AUDIT_WAVEFORM_BIN_COUNT = 560


class CompletionAuditWaveformChannel(BenchmarkModel):
    minimums: tuple[float, ...] = Field(min_length=1)
    maximums: tuple[float, ...] = Field(min_length=1)


class CompletionAuditWaveform(BenchmarkModel):
    audit_id: str
    duration_seconds: float = Field(gt=0.0)
    user: CompletionAuditWaveformChannel
    assistant: CompletionAuditWaveformChannel


class CompletionAuditAudioLoader(Protocol):
    def load(
        self,
        relative_path: str,
        start_seconds: float,
        end_seconds: float,
        sample_rate_hz: int,
    ) -> NDArray[np.float32]: ...


class CompletionAuditClipItem(Protocol):
    audit_id: str
    candidate_id: str
    clip_path: str
    clip_start_seconds: float
    clip_end_seconds: float


@dataclass(frozen=True)
class AuditReviewPageDefinition:
    kind: Literal["completion", "quality"]
    title: str
    instructions_html: str
    case_kicker: str
    question_html: str
    labels: tuple[str, ...]
    label_names: dict[str, str]
    decision_buttons_html: str
    tags: tuple[str, ...]
    notes_summary: str
    notes_placeholder: str
    metadata_summary: str
    storage_namespace: str
    review_schema_version: str
    review_filename_prefix: str


@dataclass(frozen=True)
class ResolvedCompletionAuditAudioLoader:
    path_resolver: AudioPathResolver

    def load(
        self,
        relative_path: str,
        start_seconds: float,
        end_seconds: float,
        sample_rate_hz: int,
    ) -> NDArray[np.float32]:
        waveform = load_audio_window(
            path=self.path_resolver.resolve(relative_path),
            sample_rate_hz=sample_rate_hz,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            pad_missing_suffix=True,
        )
        return waveform.numpy().astype(np.float32, copy=False)


def write_completion_audit_package(
    output_directory: Path,
    manifest: CompletionAuditManifest,
    inventory: TurnCompletionInventory,
    samples: Sequence[MaterializedTrainingSample],
    audio_loader: CompletionAuditAudioLoader,
    sample_rate_hz: int = AUDIT_SAMPLE_RATE_HZ,
    progress_output: TextIO | None = None,
) -> None:
    if sample_rate_hz <= 0:
        raise ValueError("Completion audit sample rate must be positive.")
    if inventory.manifest.candidate_sha256 != manifest.inventory_sha256:
        raise ValueError("Completion audit manifest and inventory do not match.")
    output_directory.mkdir(parents=True, exist_ok=True)
    clips_directory = output_directory / "clips"
    clips_directory.mkdir(parents=True, exist_ok=True)
    waveforms = _write_audit_clips(
        output_directory=output_directory,
        items=manifest.items,
        inventory=inventory,
        samples=samples,
        audio_loader=audio_loader,
        sample_rate_hz=sample_rate_hz,
        progress_output=progress_output,
    )

    (output_directory / "audit-manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _write_review_template(output_directory / "review-template.csv", manifest)
    (output_directory / "index.html").write_text(
        _review_page(manifest, waveforms, _completion_review_definition()),
        encoding="utf-8",
    )


def write_corpus_quality_audit_package(
    output_directory: Path,
    manifest: CorpusQualityAuditManifest,
    inventory: TurnCompletionInventory,
    samples: Sequence[MaterializedTrainingSample],
    audio_loader: CompletionAuditAudioLoader,
    sample_rate_hz: int = AUDIT_SAMPLE_RATE_HZ,
    progress_output: TextIO | None = None,
) -> None:
    if sample_rate_hz <= 0:
        raise ValueError("Corpus quality audit sample rate must be positive.")
    if inventory.manifest.candidate_sha256 != manifest.inventory_sha256:
        raise ValueError("Corpus quality audit manifest and inventory do not match.")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "clips").mkdir(parents=True, exist_ok=True)
    waveforms = _write_audit_clips(
        output_directory=output_directory,
        items=manifest.items,
        inventory=inventory,
        samples=samples,
        audio_loader=audio_loader,
        sample_rate_hz=sample_rate_hz,
        progress_output=progress_output,
    )
    (output_directory / "audit-manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_directory / "index.html").write_text(
        _review_page(manifest, waveforms, _quality_review_definition()),
        encoding="utf-8",
    )


def _write_audit_clips(
    output_directory: Path,
    items: Sequence[CompletionAuditClipItem],
    inventory: TurnCompletionInventory,
    samples: Sequence[MaterializedTrainingSample],
    audio_loader: CompletionAuditAudioLoader,
    sample_rate_hz: int,
    progress_output: TextIO | None,
) -> tuple[CompletionAuditWaveform, ...]:
    candidates_by_id = {candidate.candidate_id: candidate for candidate in inventory.candidates}
    samples_by_window_id = {sample.window_id: sample for sample in samples}
    started_at = time.perf_counter()
    if progress_output is not None:
        _write_progress(progress_output, 0, len(items), 0.0)
    waveforms: list[CompletionAuditWaveform] = []
    for completed_count, item in enumerate(items, start=1):
        candidate = candidates_by_id.get(item.candidate_id)
        if candidate is None:
            raise ValueError(f"Audit item references unknown candidate {item.candidate_id}.")
        source_sample = _source_sample(candidate.source_window_ids, samples_by_window_id)
        if source_sample.user_audio_path != candidate.user_audio_path:
            raise ValueError("Audit source window user audio does not match its candidate.")
        user_audio = audio_loader.load(
            candidate.user_audio_path,
            item.clip_start_seconds,
            item.clip_end_seconds,
            sample_rate_hz,
        )
        assistant_audio = audio_loader.load(
            source_sample.assistant_audio_path,
            item.clip_start_seconds,
            item.clip_end_seconds,
            sample_rate_hz,
        )
        _write_stereo_wave(
            output_directory / item.clip_path,
            user_audio,
            assistant_audio,
            sample_rate_hz,
        )
        waveforms.append(
            _completion_audit_waveform(
                item.audit_id,
                user_audio,
                assistant_audio,
                sample_rate_hz,
            )
        )
        if progress_output is not None:
            _write_progress(
                progress_output,
                completed_count,
                len(items),
                time.perf_counter() - started_at,
            )
    return tuple(waveforms)


def refresh_completion_audit_review_page(output_directory: Path) -> None:
    manifest_path = output_directory / "audit-manifest.json"
    manifest = CompletionAuditManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    waveforms = tuple(
        _read_completion_audit_waveform(output_directory / item.clip_path, item.audit_id)
        for item in manifest.items
    )
    (output_directory / "index.html").write_text(
        _review_page(manifest, waveforms, _completion_review_definition()),
        encoding="utf-8",
    )


def refresh_corpus_quality_audit_review_page(output_directory: Path) -> None:
    manifest_path = output_directory / "audit-manifest.json"
    manifest = CorpusQualityAuditManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    waveforms = tuple(
        _read_completion_audit_waveform(output_directory / item.clip_path, item.audit_id)
        for item in manifest.items
    )
    (output_directory / "index.html").write_text(
        _review_page(manifest, waveforms, _quality_review_definition()),
        encoding="utf-8",
    )


def _write_progress(
    output: TextIO,
    completed_count: int,
    total_count: int,
    elapsed_seconds: float,
) -> None:
    average_seconds = elapsed_seconds / completed_count if completed_count else None
    remaining_seconds = (
        (total_count - completed_count) * average_seconds if average_seconds is not None else None
    )
    eta = "?" if remaining_seconds is None else f"{remaining_seconds / 60.0:.1f} min"
    print(
        f"Completion audit clips: {completed_count}/{total_count}, ETA {eta}",
        file=output,
        flush=True,
    )


def _source_sample(
    source_window_ids: tuple[str, ...],
    samples_by_window_id: dict[str, MaterializedTrainingSample],
) -> MaterializedTrainingSample:
    matches = tuple(
        samples_by_window_id[window_id]
        for window_id in source_window_ids
        if window_id in samples_by_window_id
    )
    if not matches:
        raise ValueError("Audit candidate source windows are absent from the corpus split.")
    return min(matches, key=lambda sample: sample.window_id)


def _write_stereo_wave(
    path: Path,
    user_audio: NDArray[np.float32],
    assistant_audio: NDArray[np.float32],
    sample_rate_hz: int,
) -> None:
    if user_audio.ndim != 1 or assistant_audio.ndim != 1:
        raise ValueError("Completion audit audio tracks must be mono.")
    if user_audio.size != assistant_audio.size:
        raise ValueError("Completion audit audio tracks must have equal durations.")
    stereo = np.column_stack((user_audio, assistant_audio))
    pcm = np.clip(stereo * 32767.0, -32768.0, 32767.0).astype("<i2", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(pcm.tobytes())


def _write_review_template(path: Path, manifest: CompletionAuditManifest) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "order",
                "audit_id",
                "double_review",
                "reviewer",
                "review_label",
                "error_tags",
                "notes",
            )
        )
        for item in manifest.items:
            writer.writerow((item.order, item.audit_id, item.double_review, "", "", "", ""))


def _completion_audit_waveform(
    audit_id: str,
    user_audio: NDArray[np.float32],
    assistant_audio: NDArray[np.float32],
    sample_rate_hz: int,
) -> CompletionAuditWaveform:
    if user_audio.size != assistant_audio.size:
        raise ValueError("Completion audit waveform channels must have equal durations.")
    return CompletionAuditWaveform(
        audit_id=audit_id,
        duration_seconds=user_audio.size / sample_rate_hz,
        user=_waveform_channel(user_audio),
        assistant=_waveform_channel(assistant_audio),
    )


def _waveform_channel(samples: NDArray[np.float32]) -> CompletionAuditWaveformChannel:
    if samples.ndim != 1 or not samples.size:
        raise ValueError("Completion audit waveform input must be non-empty mono audio.")
    edge_indices = np.linspace(
        0,
        samples.size,
        min(AUDIT_WAVEFORM_BIN_COUNT, samples.size) + 1,
        dtype=np.int64,
    )
    windows = tuple(
        samples[edge_indices[index] : edge_indices[index + 1]]
        for index in range(edge_indices.size - 1)
    )
    return CompletionAuditWaveformChannel(
        minimums=tuple(float(window.min()) for window in windows),
        maximums=tuple(float(window.max()) for window in windows),
    )


def _read_completion_audit_waveform(
    clip_path: Path,
    audit_id: str,
) -> CompletionAuditWaveform:
    with wave.open(str(clip_path), "rb") as audio:
        if audio.getnchannels() != 2 or audio.getsampwidth() != 2:
            raise ValueError("Completion audit review clips must be 16-bit stereo WAV files.")
        sample_rate_hz = audio.getframerate()
        sample_count = audio.getnframes()
        stereo = np.frombuffer(audio.readframes(sample_count), dtype="<i2").reshape(-1, 2)
    normalized = stereo.astype(np.float32) / 32768.0
    return _completion_audit_waveform(
        audit_id,
        normalized[:, 0],
        normalized[:, 1],
        sample_rate_hz,
    )


def _completion_review_definition() -> AuditReviewPageDefinition:
    return AuditReviewPageDefinition(
        kind="completion",
        title="Voice Light completion-label audit",
        instructions_html="""
      <p><strong>There is one decision point per clip: the vertical pink line at t = 0.</strong>
      Label whether the candidate user has completed their turn at that exact instant.</p>
      <div class="instruction-grid">
        <div><strong>SAFE TO TAKE</strong>The assistant can begin at the line without cutting off
        the candidate user.</div>
        <div><strong>HOLD</strong>The candidate user is pausing mid-turn; beginning at the line
        would interrupt their continuation.</div>
        <div><strong>AMBIGUOUS</strong>The audio, timing, overlap, or wording does not support a
        reliable binary judgment.</div>
      </div>
      <p class="muted">Use audio after the line to verify what followed, but do not move the
      decision point. The top waveform is always the candidate user; the bottom is the other
      speaker.</p>""",
        case_kicker="Completion boundary review",
        question_html="""<strong>At the pink line:</strong> can the assistant start speaking
      without cutting off the candidate user?""",
        labels=("safe_to_take", "hold", "ambiguous_unratable"),
        label_names={
            "safe_to_take": "SAFE TO TAKE",
            "hold": "HOLD",
            "ambiguous_unratable": "AMBIGUOUS / UNRATABLE",
        },
        decision_buttons_html="""
        <button data-label="safe_to_take" aria-pressed="false"><kbd>1</kbd>
          <strong>Safe to take</strong>
          <span>The candidate user's turn is complete at the line.</span></button>
        <button data-label="hold" aria-pressed="false"><kbd>2</kbd>
          <strong>Hold</strong>
          <span>The candidate user still owns the turn at the line.</span></button>
        <button data-label="ambiguous_unratable" aria-pressed="false"><kbd>3</kbd>
          <strong>Ambiguous / unratable</strong>
          <span>A reliable binary label is not justified.</span></button>""",
        tags=(
            "continuation",
            "backchannel",
            "overlap",
            "transcript_error",
            "timing_error",
            "censored_context",
            "audio_quality",
        ),
        notes_summary="Optional error tags and notes",
        notes_placeholder="Why is this label questionable?",
        metadata_summary="Reveal automatic label, selection reasons, and model scores",
        storage_namespace="completion-audit",
        review_schema_version="voice-light-completion-label-reviews-v1",
        review_filename_prefix="completion-label-reviews",
    )


def _quality_review_definition() -> AuditReviewPageDefinition:
    return AuditReviewPageDefinition(
        kind="quality",
        title="Voice Light corpus-quality control audit",
        instructions_html="""
      <p><strong>This is a 20-case go/no-go check, not a completion-labeling task.</strong>
      Judge whether each clip is suitable training material and whether the marked boundary and
      channel roles look correctly extracted.</p>
      <div class="instruction-grid">
        <div><strong>USABLE</strong>A coherent two-party exchange with intelligible audio,
        plausible channel roles, and a boundary in the expected conversational location.</div>
        <div><strong>BAD SOURCE</strong>The recording or conversation itself is too degraded,
        fragmented, or unnatural to be useful training material.</div>
        <div><strong>WRONG ALIGNMENT / CHANNEL</strong>The underlying audio may be usable, but the
        pink boundary, timing, or candidate/other-speaker channel assignment looks wrong.</div>
        <div><strong>UNSURE</strong>You cannot reliably distinguish source quality from an
        extraction problem.</div>
      </div>
      <p class="muted">Do not decide HOLD versus end-of-turn. The top waveform is the candidate
      user channel and the bottom is the other speaker. Use the pink line only to check whether
      extraction and conversational timing are plausible.</p>""",
        case_kicker="Corpus quality control",
        question_html="""<strong>Is this usable training material?</strong> Separate an
      inherently bad recording/conversation from a likely timing or channel extraction error.""",
        labels=(
            "usable",
            "unusable_source",
            "wrong_alignment_or_channel",
            "unsure",
        ),
        label_names={
            "usable": "USABLE",
            "unusable_source": "BAD SOURCE",
            "wrong_alignment_or_channel": "WRONG ALIGNMENT / CHANNEL",
            "unsure": "UNSURE",
        },
        decision_buttons_html="""
        <button data-label="usable" aria-pressed="false"><kbd>1</kbd>
          <strong>Usable</strong>
          <span>Coherent audio, plausible channels, and plausible boundary placement.</span>
        </button>
        <button data-label="unusable_source" aria-pressed="false"><kbd>2</kbd>
          <strong>Bad source</strong>
          <span>The original recording or conversation is not useful training material.</span>
        </button>
        <button data-label="wrong_alignment_or_channel" aria-pressed="false"><kbd>3</kbd>
          <strong>Wrong alignment / channel</strong>
          <span>The timing, boundary, or speaker-channel assignment appears incorrect.</span>
        </button>
        <button data-label="unsure" aria-pressed="false"><kbd>4</kbd>
          <strong>Unsure</strong>
          <span>The cause of the problem cannot be determined reliably.</span></button>""",
        tags=(),
        notes_summary="Optional note",
        notes_placeholder="Only add a note if it helps explain the failure.",
        metadata_summary="Reveal source and candidate metadata",
        storage_namespace="corpus-quality-audit",
        review_schema_version="voice-light-corpus-quality-reviews-v1",
        review_filename_prefix="corpus-quality-reviews",
    )


def _review_page(
    manifest: CompletionAuditManifest | CorpusQualityAuditManifest,
    waveforms: tuple[CompletionAuditWaveform, ...],
    definition: AuditReviewPageDefinition,
) -> str:
    if tuple(waveform.audit_id for waveform in waveforms) != tuple(
        item.audit_id for item in manifest.items
    ):
        raise ValueError("Completion audit waveform order does not match its manifest.")
    manifest_json = json.dumps(manifest.model_dump(mode="json")).replace("<", "\\u003c")
    waveform_json = json.dumps(
        tuple(waveform.model_dump(mode="json") for waveform in waveforms),
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    title = html.escape(definition.title)
    labels_json = json.dumps(definition.labels, separators=(",", ":"))
    label_names_json = json.dumps(definition.label_names, separators=(",", ":"))
    tags_json = json.dumps(definition.tags, separators=(",", ":"))
    kind_json = json.dumps(definition.kind)
    storage_namespace_json = json.dumps(definition.storage_namespace)
    review_schema_version_json = json.dumps(definition.review_schema_version)
    review_filename_prefix_json = json.dumps(definition.review_filename_prefix)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      background: #0b1020; color: #e8edf7; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0 auto; max-width: 1180px; padding: 28px; line-height: 1.45; }}
    header, main {{ display: grid; gap: 18px; }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: clamp(24px, 4vw, 38px); letter-spacing: -0.03em; }}
    h2 {{ font-size: 20px; }}
    .card {{ background: #121a2d; border: 1px solid #2b3855; border-radius: 16px;
      box-shadow: 0 12px 38px #0005; padding: 22px; }}
    .instructions {{ display: grid; gap: 12px; border-left: 5px solid #fb7185; }}
    .instruction-grid {{ display: grid; gap: 10px;
      grid-template-columns: repeat({len(definition.labels)}, 1fr); }}
    .instruction-grid div {{ background: #0d1425; border-radius: 10px; padding: 12px; }}
    .instruction-grid strong {{ color: #fda4af; display: block; margin-bottom: 4px; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .spread {{ justify-content: space-between; }}
    button, input, textarea {{ font: inherit; }}
    button {{ background: #1b2944; border: 1px solid #3b4e73; border-radius: 9px;
      color: #eef4ff; cursor: pointer; padding: 9px 13px; }}
    button:hover {{ background: #243655; }}
    button:focus-visible, input:focus-visible, textarea:focus-visible {{
      outline: 3px solid #38bdf8; }}
    button.selected {{ border-color: #34d399; box-shadow: 0 0 0 3px #34d39955; }}
    button.reviewer-disagrees {{ border-color: #fb7185; box-shadow: 0 0 0 3px #fb718555; }}
    input, textarea {{ background: #0b1222; border: 1px solid #3b4e73; border-radius: 8px;
      color: #eef4ff; padding: 8px; }}
    .bar {{ height: 9px; background: #26334d; border-radius: 8px; overflow: hidden; }}
    .bar > div {{ height: 100%; background: #34d399; transition: width .2s; }}
    .case-card {{ margin-top: 20px; }}
    .case-kicker {{ color: #94a3b8; font-size: 13px; font-weight: 700; letter-spacing: .09em;
      text-transform: uppercase; }}
    .question {{ background: #231725; border: 1px solid #713448; border-radius: 12px;
      font-size: 18px; padding: 15px; }}
    .question strong {{ color: #fda4af; }}
    .feedback {{ align-items: center; background: #0d1728; border: 1px solid #3b4e73;
      border-radius: 12px; display: flex; flex-wrap: wrap; font-weight: 700; gap: 10px;
      min-height: 52px; padding: 10px 14px; }}
    .feedback.hidden {{ display: none; }}
    .feedback-chip {{ border-radius: 7px; color: #071016; padding: 6px 9px; }}
    .feedback-original {{ background: #4ade80; }}
    .feedback-reviewer {{ background: #fb7185; }}
    .waveform-stack {{ border: 1px solid #334565; border-radius: 13px; overflow: hidden; }}
    .waveform-panel {{ background: #091121; padding: 10px 12px 12px; }}
    .waveform-panel + .waveform-panel {{ border-top: 1px solid #334565; }}
    .channel-label {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
    .channel-label strong {{ color: #dbeafe; }}
    .channel-label span {{ color: #94a3b8; font-size: 13px; }}
    canvas {{ cursor: crosshair; display: block; height: 132px; width: 100%; }}
    .boundary-key {{ align-items: center; color: #cbd5e1; display: flex; font-size: 13px;
      gap: 8px; margin-top: 10px; }}
    .boundary-swatch {{ background: #fb7185; display: inline-block; height: 18px; width: 3px; }}
    audio {{ width: 100%; }}
    .playback {{ display: grid; gap: 10px; }}
    .autoplay-gate {{ background: #2563eb; border-color: #60a5fa; font-weight: 800; }}
    .autoplay-gate.hidden {{ display: none; }}
    .decision-grid {{ display: grid; gap: 10px; grid-template-columns: repeat(3, 1fr); }}
    .decision-grid button {{ min-height: 112px; text-align: left; }}
    .decision-grid strong {{ display: block; font-size: 17px; margin: 6px 0; }}
    .decision-grid span {{ color: #bdc8da; display: block; font-size: 13px; }}
    textarea {{ min-height: 78px; resize: vertical; width: 100%; }}
    .tags label {{ display: inline-block; margin: 5px 14px 5px 0; }}
    .muted {{ color: #94a3b8; }}
    details {{ border-top: 1px solid #334565; padding-top: 12px; }}
    details summary {{ cursor: pointer; }}
    pre {{ overflow-x: auto; white-space: pre-wrap; }}
    kbd {{ border: 1px solid #64748b; border-radius: 4px; padding: 1px 5px; }}
    @media (max-width: 760px) {{
      body {{ padding: 14px; }}
      .instruction-grid, .decision-grid {{ grid-template-columns: 1fr; }}
      canvas {{ height: 108px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <section class="card instructions" aria-labelledby="instructions-heading">
      <h2 id="instructions-heading">What you are annotating</h2>
      {definition.instructions_html}
    </section>
    <div class="row spread">
      <div class="row">
        <label>Reviewer <input id="reviewer" autocomplete="name" placeholder="Your name"></label>
        <span id="progress"></span>
      </div>
      <button id="export">Export reviews JSON</button>
    </div>
    <div class="bar"><div id="progress-bar"></div></div>
  </header>
  <main class="card case-card">
    <div class="row spread">
      <div>
        <div class="case-kicker">{html.escape(definition.case_kicker)}</div>
        <strong id="position"></strong>
        <span id="double-review" class="muted"></span>
      </div>
      <div class="row">
        <button id="previous">← Previous</button>
        <button id="next">Next →</button>
      </div>
    </div>
    <div class="question">{definition.question_html}</div>
    <div id="feedback" class="feedback hidden" role="status" aria-live="polite"></div>
    <section aria-label="Synchronized two-channel waveform">
      <div class="waveform-stack">
        <div class="waveform-panel">
          <div class="channel-label"><strong>Candidate user</strong>
            <span>audio left channel</span></div>
          <canvas id="user-waveform" aria-label="Candidate user waveform"></canvas>
        </div>
        <div class="waveform-panel">
          <div class="channel-label"><strong>Other speaker</strong>
            <span>audio right channel</span></div>
          <canvas id="assistant-waveform" aria-label="Other speaker waveform"></canvas>
        </div>
      </div>
      <div class="boundary-key"><span class="boundary-swatch"></span>
        <strong>Pink line = annotation boundary (t = 0)</strong>
        <span id="boundary" class="muted"></span>
      </div>
    </section>
    <section class="playback" aria-label="Playback controls">
      <audio id="audio" controls preload="auto"></audio>
      <button id="enable-autoplay" class="autoplay-gate hidden">
        Start review and enable autoplay</button>
      <div class="row">
        <button id="play-full">Play full clip</button>
        <button id="play-before">Play −3 s to boundary</button>
        <button id="play-around">Play −2 s to +2 s</button>
        <span class="muted">Click either waveform to seek.</span>
        <span id="autoplay-status" class="muted"></span>
      </div>
    </section>
    <section aria-labelledby="decision-heading">
      <h2 id="decision-heading">Your label</h2>
      <div class="decision-grid" id="labels">
        {definition.decision_buttons_html}
      </div>
    </section>
    <details>
      <summary>{html.escape(definition.notes_summary)}</summary>
      <div class="tags" id="tags"></div>
      <label>Notes
        <textarea id="notes" placeholder="{html.escape(definition.notes_placeholder)}"></textarea>
      </label>
    </details>
    <details id="metadata">
      <summary>{html.escape(definition.metadata_summary)}</summary>
      <pre id="metadata-text"></pre>
    </details>
  </main>
  <script>
    const manifest = {manifest_json};
    const waveformRows = {waveform_json};
    const reviewerKind = {kind_json};
    const waveformById = Object.fromEntries(waveformRows.map(row => [row.audit_id, row]));
    const labels = {labels_json};
    const labelNames = {label_names_json};
    const tags = {tags_json};
    const storageNamespace = {storage_namespace_json};
    const reviewSchemaVersion = {review_schema_version_json};
    const reviewFilenamePrefix = {review_filename_prefix_json};
    const storagePrefix = `${{storageNamespace}}:${{manifest.items_sha256}}`;
    let state = {{index:0, reviews:{{}}}};
    let playbackEnd = null;
    let animationFrame = null;
    let feedback = null;
    let feedbackTimer = null;
    let waveformPeak = 0.01;
    let autoplayAttempt = 0;
    const byId = id => document.getElementById(id);
    const audio = byId("audio");
    const canvases = [byId("user-waveform"), byId("assistant-waveform")];

    function current() {{ return manifest.items[state.index]; }}
    function currentWaveform() {{ return waveformById[current().audit_id]; }}
    function reviewerName() {{ return byId("reviewer").value.trim(); }}
    function save() {{
      if (reviewerName()) localStorage.setItem(`${{storagePrefix}}:${{reviewerName()}}`,
        JSON.stringify(state));
    }}
    function loadReviewer() {{
      state = JSON.parse(localStorage.getItem(`${{storagePrefix}}:${{reviewerName()}}`) ||
        '{{"index":0,"reviews":{{}}}}');
      render();
    }}
    function review() {{
      const item = current();
      return state.reviews[item.audit_id] ||= {{review_label:null, error_tags:[], notes:""}};
    }}
    function originalLabel(item) {{
      if (reviewerKind !== "completion") return null;
      if (item.group === "confident_eot") return "safe_to_take";
      if (item.group === "confident_hold") return "hold";
      return "ambiguous_unratable";
    }}
    function displayLabel(value) {{
      return labelNames[value];
    }}
    function setLabel(value) {{
      if (feedback !== null) return;
      const item = current();
      review().review_label = value; save(); audio.pause(); autoplayAttempt += 1;
      byId("enable-autoplay").classList.add("hidden");
      feedback = {{auditId:item.audit_id, original:originalLabel(item), reviewer:value}};
      renderDecision(); renderFeedback(); drawWaveforms();
      feedbackTimer = setTimeout(() => {{
        if (state.index < manifest.items.length - 1) move(1);
        else {{ feedback = null; renderDecision(); renderFeedback(); drawWaveforms(); }}
      }}, 1000);
    }}
    function move(delta) {{
      clearTimeout(feedbackTimer); feedback = null;
      state.index = Math.max(0, Math.min(manifest.items.length - 1, state.index + delta));
      playbackEnd = null; audio.pause(); save(); render();
    }}
    function renderFeedback() {{
      const panel = byId("feedback");
      if (feedback === null || feedback.auditId !== current().audit_id) {{
        panel.classList.add("hidden"); panel.replaceChildren(); return;
      }}
      const matches = feedback.original === feedback.reviewer;
      panel.classList.remove("hidden");
      panel.innerHTML = feedback.original === null
        ? `<span class="feedback-chip feedback-original">RECORDED</span>
          ${{displayLabel(feedback.reviewer)}} — advancing in one second`
        : matches
        ? `<span class="feedback-chip feedback-original">AGREEMENT</span>
          Original and your label: ${{displayLabel(feedback.original)}}`
        : `<span class="feedback-chip feedback-original">ORIGINAL ·
          ${{displayLabel(feedback.original)}}</span>
          <span class="feedback-chip feedback-reviewer">YOUR LABEL ·
          ${{displayLabel(feedback.reviewer)}}</span>
          <span>Disagreement — advancing in one second</span>`;
    }}
    function renderDecision() {{
      const value = review();
      document.querySelectorAll("[data-label]").forEach(button => {{
        const selected = button.dataset.label === value.review_label;
        const disagrees = selected && feedback?.original !== null &&
          feedback.original !== feedback.reviewer;
        button.classList.toggle("selected", selected);
        button.classList.toggle("reviewer-disagrees", disagrees);
        button.setAttribute("aria-pressed", String(selected));
        button.disabled = feedback !== null;
      }});
      const completed = Object.values(state.reviews).filter(item => item.review_label).length;
      byId("progress").textContent = `${{completed}} / ${{manifest.item_count}} reviewed`;
      byId("progress-bar").style.width = `${{100 * completed / manifest.item_count}}%`;
    }}
    function render() {{
      const item = current(); const value = review();
      const waveform = currentWaveform();
      waveformPeak = Math.max(0.01,
        ...waveform.user.minimums.map(Math.abs), ...waveform.user.maximums.map(Math.abs),
        ...waveform.assistant.minimums.map(Math.abs),
        ...waveform.assistant.maximums.map(Math.abs));
      byId("position").textContent = `Case ${{item.order}} of ${{manifest.item_count}}`;
      byId("double-review").textContent = reviewerKind === "completion" && item.double_review
        ? " · independent double review" : "";
      audio.oncanplay = null;
      audio.oncanplay = () => {{
        audio.oncanplay = null;
        attemptAutoplay(item.audit_id);
      }};
      audio.src = item.clip_path;
      audio.load();
      byId("boundary").textContent = `(${{item.boundary_offset_seconds.toFixed(2)}} s into clip)`;
      byId("tags").innerHTML = tags.map(tag =>
        `<label><input type="checkbox" value="${{tag}}"
        ${{value.error_tags.includes(tag) ? "checked" : ""}}>
        ${{tag.replaceAll("_", " ")}}</label>`).join("");
      byId("tags").querySelectorAll("input").forEach(input => input.onchange = () => {{
        value.error_tags = [...byId("tags").querySelectorAll("input:checked")]
          .map(node => node.value);
        save();
      }});
      byId("notes").value = value.notes;
      byId("metadata").open = false;
      const metadata = {{
        boundary_kind: item.boundary_kind,
        dataset: item.dataset_name,
        categories: item.categories,
        candidate_id: item.candidate_id
      }};
      if (reviewerKind === "completion") Object.assign(metadata, {{
        automatic_group: item.group,
        completion_probability: item.completion_probability,
        continuation_probability: item.continuation_probability,
        selection_reasons: item.selection_reasons,
        challenge_score: item.challenge_score,
        voice_light: item.voice_light,
        smart_turn: item.smart_turn,
        livekit: item.livekit
      }});
      byId("metadata-text").textContent = JSON.stringify(metadata, null, 2);
      renderDecision(); renderFeedback();
      requestAnimationFrame(drawWaveforms);
    }}

    function attemptAutoplay(auditId) {{
      if (current().audit_id !== auditId || feedback !== null) return;
      const attempt = ++autoplayAttempt;
      playbackEnd = null;
      audio.play().then(() => {{
        if (attempt !== autoplayAttempt || current().audit_id !== auditId) return;
        byId("enable-autoplay").classList.add("hidden");
        byId("autoplay-status").textContent = "Autoplaying full clip";
      }}).catch(() => {{
        if (attempt !== autoplayAttempt || current().audit_id !== auditId) return;
        byId("enable-autoplay").classList.remove("hidden");
        byId("autoplay-status").textContent =
          "The browser requires one click before it permits sound.";
      }});
    }}

    function drawWaveforms() {{
      const waveform = currentWaveform();
      drawChannel(canvases[0], waveform.user, waveform, waveformPeak, "#38bdf8", true);
      drawChannel(canvases[1], waveform.assistant, waveform, waveformPeak, "#a78bfa", false);
    }}
    function drawChannel(canvas, channel, waveform, peak, color, labelBoundary) {{
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, canvas.clientWidth);
      const height = Math.max(1, canvas.clientHeight);
      const pixelWidth = Math.round(width * ratio); const pixelHeight = Math.round(height * ratio);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {{
        canvas.width = pixelWidth; canvas.height = pixelHeight;
      }}
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      const item = current();
      const boundaryX = width * item.boundary_offset_seconds / waveform.duration_seconds;
      context.fillStyle = "#38bdf80b"; context.fillRect(0, 0, boundaryX, height);
      context.fillStyle = "#fb923c0d"; context.fillRect(boundaryX, 0, width - boundaryX, height);
      context.font = "11px system-ui"; context.textAlign = "center";
      for (let second = Math.ceil(-item.boundary_offset_seconds);
        second <= Math.floor(waveform.duration_seconds - item.boundary_offset_seconds); second++) {{
        const x = width * (second + item.boundary_offset_seconds) / waveform.duration_seconds;
        context.strokeStyle = second === 0 ? "#fb7185" : "#334155";
        context.lineWidth = second === 0 ? 3 : 1;
        context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
        if (second !== 0) {{
          context.fillStyle = "#8290a7";
          context.textAlign = x < 24 ? "left" : (x > width - 24 ? "right" : "center");
          context.fillText(`${{second > 0 ? "+" : ""}}${{second}}s`, x, height - 6);
        }}
      }}
      context.strokeStyle = color; context.fillStyle = `${{color}}35`; context.lineWidth = 1;
      context.beginPath();
      const center = height / 2; const amplitude = height * 0.40 / peak;
      for (let index = 0; index < channel.maximums.length; index++) {{
        const x = width * index / Math.max(1, channel.maximums.length - 1);
        const y = center - channel.maximums[index] * amplitude;
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      }}
      for (let index = channel.minimums.length - 1; index >= 0; index--) {{
        const x = width * index / Math.max(1, channel.minimums.length - 1);
        context.lineTo(x, center - channel.minimums[index] * amplitude);
      }}
      context.closePath(); context.fill(); context.stroke();
      const activeFeedback = feedback?.auditId === item.audit_id ? feedback : null;
      if (activeFeedback === null) {{
        drawBoundary(context, boundaryX, "#fb7185", 3, height);
      }} else if (activeFeedback.original === null ||
        activeFeedback.original === activeFeedback.reviewer) {{
        drawBoundary(context, boundaryX, "#4ade80", 5, height);
      }} else {{
        drawBoundary(context, boundaryX - 3, "#4ade80", 4, height);
        drawBoundary(context, boundaryX + 3, "#fb7185", 4, height);
      }}
      if (labelBoundary) {{
        context.fillStyle = "#8290a7"; context.font = "11px system-ui";
        context.textAlign = "left"; context.fillText("BEFORE BOUNDARY", 8, 16);
        context.textAlign = "right"; context.fillText("AFTER · OUTCOME CONTEXT", width - 8, 16);
        context.font = "bold 12px system-ui";
        if (activeFeedback === null) {{
          context.fillStyle = "#fb7185";
          context.textAlign = boundaryX > width * .75 ? "right" : "left";
          context.fillText("DECISION POINT · t=0",
            boundaryX + (boundaryX > width * .75 ? -8 : 8), 34);
        }} else if (activeFeedback.original === null) {{
          context.fillStyle = "#4ade80"; context.textAlign = "left";
          context.fillText(`RECORDED · ${{displayLabel(activeFeedback.reviewer)}}`,
            boundaryX + 9, 34);
        }} else if (activeFeedback.original === activeFeedback.reviewer) {{
          context.fillStyle = "#4ade80"; context.textAlign = "left";
          context.fillText(`AGREEMENT · ${{displayLabel(activeFeedback.original)}}`,
            boundaryX + 9, 34);
        }} else {{
          context.fillStyle = "#4ade80"; context.textAlign = "right";
          context.fillText(`ORIGINAL · ${{displayLabel(activeFeedback.original)}}`,
            boundaryX - 10, 34);
          context.fillStyle = "#fb7185"; context.textAlign = "left";
          context.fillText(`YOURS · ${{displayLabel(activeFeedback.reviewer)}}`,
            boundaryX + 10, 52);
        }}
      }}
      if (Number.isFinite(audio.currentTime)) {{
        const playheadX = width * audio.currentTime / waveform.duration_seconds;
        context.strokeStyle = "#f8fafc"; context.lineWidth = 1;
        context.beginPath(); context.moveTo(playheadX, 0);
        context.lineTo(playheadX, height); context.stroke();
      }}
    }}
    function drawBoundary(context, x, color, width, height) {{
      context.strokeStyle = color; context.lineWidth = width;
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
    }}
    function seekFromPointer(event) {{
      const rectangle = event.currentTarget.getBoundingClientRect();
      audio.currentTime = currentWaveform().duration_seconds *
        Math.max(0, Math.min(1, (event.clientX - rectangle.left) / rectangle.width));
      drawWaveforms();
    }}
    function playRange(start, end) {{
      playbackEnd = end; audio.currentTime = Math.max(0, start); audio.play();
    }}
    function animatePlayback() {{
      drawWaveforms();
      if (!audio.paused) animationFrame = requestAnimationFrame(animatePlayback);
    }}

    document.querySelectorAll("[data-label]").forEach(button =>
      button.onclick = () => setLabel(button.dataset.label));
    canvases.forEach(canvas => canvas.onpointerdown = seekFromPointer);
    byId("previous").onclick = () => move(-1); byId("next").onclick = () => move(1);
    byId("notes").oninput = event => {{ review().notes = event.target.value; save(); }};
    byId("reviewer").onchange = loadReviewer;
    byId("enable-autoplay").onclick = () => {{
      autoplayAttempt += 1;
      byId("enable-autoplay").classList.add("hidden");
      byId("autoplay-status").textContent = "Autoplay enabled";
      playRange(0, null);
    }};
    byId("play-full").onclick = () => playRange(0, null);
    byId("play-before").onclick = () => {{
      const boundary = current().boundary_offset_seconds;
      playRange(boundary - 3, boundary);
    }};
    byId("play-around").onclick = () => {{
      const boundary = current().boundary_offset_seconds;
      playRange(boundary - 2, Math.min(currentWaveform().duration_seconds, boundary + 2));
    }};
    audio.ontimeupdate = () => {{
      if (playbackEnd !== null && audio.currentTime >= playbackEnd) {{
        audio.pause(); playbackEnd = null;
      }}
      drawWaveforms();
    }};
    audio.onplay = () => {{ cancelAnimationFrame(animationFrame); animatePlayback(); }};
    audio.onpause = () => {{ cancelAnimationFrame(animationFrame); drawWaveforms(); }};
    byId("export").onclick = () => {{
      if (!reviewerName()) {{ alert("Enter a reviewer name before exporting."); return; }}
      const payload = {{schema_version:reviewSchemaVersion,
        manifest_sha256:manifest.items_sha256, reviewer:reviewerName(),
        reviews:manifest.items.filter(item => state.reviews[item.audit_id]?.review_label)
          .map(item => reviewerKind === "completion"
            ? ({{audit_id:item.audit_id, ...state.reviews[item.audit_id]}})
            : ({{audit_id:item.audit_id,
              review_label:state.reviews[item.audit_id].review_label,
              notes:state.reviews[item.audit_id].notes}}))}};
      const link = document.createElement("a");
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type:"application/json"}});
      link.href = URL.createObjectURL(blob);
      link.download = `${{reviewFilenamePrefix}}-${{payload.reviewer}}.json`;
      link.click(); URL.revokeObjectURL(link.href);
    }};
    document.onkeydown = event => {{
      if (event.target.matches("input, textarea")) return;
      if (labels[Number(event.key) - 1]) setLabel(labels[Number(event.key) - 1]);
      if (event.key === "ArrowRight") move(1); if (event.key === "ArrowLeft") move(-1);
      if (event.key === " ") {{
        event.preventDefault(); audio.paused ? audio.play() : audio.pause();
      }}
    }};
    new ResizeObserver(drawWaveforms).observe(byId("user-waveform"));
    render();
  </script>
</body>
</html>
"""
