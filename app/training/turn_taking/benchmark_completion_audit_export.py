from __future__ import annotations

import csv
import html
import json
import time
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

import numpy as np
from numpy.typing import NDArray

from app.local.training_corpus.export import MaterializedTrainingSample
from app.training.turn_taking.benchmark_adapters import AudioPathResolver
from app.training.turn_taking.benchmark_completion_audit import CompletionAuditManifest
from app.training.turn_taking.benchmark_models import TurnCompletionInventory
from app.training.turn_taking.data import load_audio_window

AUDIT_SAMPLE_RATE_HZ = 16_000


class CompletionAuditAudioLoader(Protocol):
    def load(
        self,
        relative_path: str,
        start_seconds: float,
        end_seconds: float,
        sample_rate_hz: int,
    ) -> NDArray[np.float32]: ...


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
    candidates_by_id = {candidate.candidate_id: candidate for candidate in inventory.candidates}
    samples_by_window_id = {sample.window_id: sample for sample in samples}
    output_directory.mkdir(parents=True, exist_ok=True)
    clips_directory = output_directory / "clips"
    clips_directory.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    if progress_output is not None:
        _write_progress(progress_output, 0, manifest.item_count, 0.0)
    for completed_count, item in enumerate(manifest.items, start=1):
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
        if progress_output is not None:
            _write_progress(
                progress_output,
                completed_count,
                manifest.item_count,
                time.perf_counter() - started_at,
            )

    (output_directory / "audit-manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _write_review_template(output_directory / "review-template.csv", manifest)
    (output_directory / "index.html").write_text(
        _review_page(manifest),
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


def _review_page(manifest: CompletionAuditManifest) -> str:
    manifest_json = json.dumps(manifest.model_dump(mode="json")).replace("<", "\\u003c")
    title = html.escape("Voice Light completion-label audit")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0 auto; max-width: 900px; padding: 24px; line-height: 1.45; }}
    header, main {{ display: grid; gap: 16px; }}
    .bar {{ height: 10px; background: #7774; border-radius: 8px; overflow: hidden; }}
    .bar > div {{ height: 100%; background: #2c8; transition: width .2s; }}
    .card {{ border: 1px solid #8886; border-radius: 12px; padding: 20px; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    button, input, textarea {{ font: inherit; }}
    button {{ padding: 9px 13px; }}
    button.selected {{ outline: 3px solid #2c8; }}
    audio {{ width: 100%; }}
    textarea {{ box-sizing: border-box; min-height: 80px; width: 100%; }}
    .tags label {{ display: inline-block; margin: 4px 12px 4px 0; }}
    .muted {{ color: #888; }}
    details {{ border-top: 1px solid #8884; margin-top: 14px; padding-top: 10px; }}
    kbd {{ border: 1px solid #8888; border-radius: 4px; padding: 1px 5px; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>Listen blind before revealing metadata. Left channel is the candidate user; right channel
    is the other speaker. Decide whether an assistant could safely take the turn at the marked
    boundary.</p>
    <div class="row">
      <label>Reviewer <input id="reviewer" autocomplete="name"></label>
      <button id="export">Export reviews JSON</button>
      <span id="progress"></span>
    </div>
    <div class="bar"><div id="progress-bar"></div></div>
  </header>
  <main class="card">
    <div class="row">
      <button id="previous">Previous</button>
      <button id="next">Next</button>
      <strong id="position"></strong>
      <span id="double-review" class="muted"></span>
    </div>
    <p id="context"></p>
    <audio id="audio" controls preload="metadata"></audio>
    <div class="row">
      <button id="play-boundary">Play from 2 s before boundary</button>
      <span id="boundary"></span>
    </div>
    <h2>Decision</h2>
    <div class="row" id="labels">
      <button data-label="safe_to_take"><kbd>1</kbd> Safe to take</button>
      <button data-label="hold"><kbd>2</kbd> Hold</button>
      <button data-label="ambiguous_unratable"><kbd>3</kbd> Ambiguous / unratable</button>
    </div>
    <div class="tags" id="tags"></div>
    <label>Notes<textarea id="notes"></textarea></label>
    <details id="metadata">
      <summary>Reveal automatic label, selection reasons, and model scores</summary>
      <pre id="metadata-text"></pre>
    </details>
  </main>
  <script>
    const manifest = {manifest_json};
    const labels = ["safe_to_take", "hold", "ambiguous_unratable"];
    const tags = ["continuation", "backchannel", "overlap", "transcript_error",
      "timing_error", "censored_context", "audio_quality"];
    const storagePrefix = `completion-audit:${{manifest.items_sha256}}`;
    let state = {{index:0, reviews:{{}}}};
    const byId = id => document.getElementById(id);

    function current() {{ return manifest.items[state.index]; }}
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
    function setLabel(value) {{ review().review_label = value; save(); render(); }}
    function move(delta) {{
      state.index = Math.max(0, Math.min(manifest.items.length - 1, state.index + delta));
      save(); render();
    }}
    function render() {{
      const item = current(); const value = review();
      byId("position").textContent = `${{item.order}} / ${{manifest.item_count}}`;
      byId("double-review").textContent = item.double_review ? "Independent double review" : "";
      byId("context").textContent = `${{item.dataset_name}} · ${{item.categories.join(", ")}}`;
      byId("audio").src = item.clip_path;
      byId("boundary").textContent = `Boundary at ${{item.boundary_offset_seconds.toFixed(2)}} s`;
      document.querySelectorAll("[data-label]").forEach(button =>
        button.classList.toggle("selected", button.dataset.label === value.review_label));
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
      byId("metadata-text").textContent = JSON.stringify({{
        automatic_group: item.group,
        completion_probability: item.completion_probability,
        continuation_probability: item.continuation_probability,
        boundary_kind: item.boundary_kind,
        selection_reasons: item.selection_reasons,
        challenge_score: item.challenge_score,
        voice_light: item.voice_light,
        smart_turn: item.smart_turn,
        livekit: item.livekit,
        candidate_id: item.candidate_id
      }}, null, 2);
      const completed = Object.values(state.reviews).filter(item => item.review_label).length;
      byId("progress").textContent = `${{completed}} / ${{manifest.item_count}} reviewed`;
      byId("progress-bar").style.width = `${{100 * completed / manifest.item_count}}%`;
    }}

    document.querySelectorAll("[data-label]").forEach(button =>
      button.onclick = () => setLabel(button.dataset.label));
    byId("previous").onclick = () => move(-1); byId("next").onclick = () => move(1);
    byId("notes").oninput = event => {{ review().notes = event.target.value; save(); }};
    byId("reviewer").onchange = loadReviewer;
    byId("play-boundary").onclick = () => {{
      const audio = byId("audio");
      audio.currentTime = Math.max(0, current().boundary_offset_seconds - 2);
      audio.play();
    }};
    byId("export").onclick = () => {{
      if (!reviewerName()) {{ alert("Enter a reviewer name before exporting."); return; }}
      const payload = {{schema_version:"voice-light-completion-label-reviews-v1",
        manifest_sha256:manifest.items_sha256, reviewer:reviewerName(),
        reviews:manifest.items.filter(item => state.reviews[item.audit_id]?.review_label)
          .map(item => ({{audit_id:item.audit_id, ...state.reviews[item.audit_id]}}))}};
      const link = document.createElement("a");
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type:"application/json"}});
      link.href = URL.createObjectURL(blob);
      link.download = `completion-label-reviews-${{payload.reviewer || "anonymous"}}.json`;
      link.click();
      URL.revokeObjectURL(link.href);
    }};
    document.onkeydown = event => {{
      if (event.target.matches("input, textarea")) return;
      if (["1", "2", "3"].includes(event.key)) setLabel(labels[Number(event.key) - 1]);
      if (event.key === "ArrowRight") move(1); if (event.key === "ArrowLeft") move(-1);
    }};
    render();
  </script>
</body>
</html>
"""
