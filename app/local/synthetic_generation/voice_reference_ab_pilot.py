from __future__ import annotations

import hashlib
import html
import os
import shutil
import wave
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from app.local.synthetic_generation.completion_dataset import (
    SilenceDetectionConfiguration,
)
from app.local.synthetic_generation.conversation_compiler import MeasuredSilence
from app.local.synthetic_generation.conversation_tts import measure_internal_silences
from app.local.synthetic_generation.conversation_voice_references import (
    TtsBackendIdentity,
    file_sha256,
    trim_generated_speech,
    write_pcm16_wave,
)
from app.local.synthetic_generation.models import SyntheticModel

REFERENCE_TEXT = (
    "I was running late this morning, so I grabbed my coffee and hurried to the station "
    "before the meeting."
)
AUDITION_SILENCE_DETECTION = SilenceDetectionConfiguration(
    absolute_rms_threshold=0.003,
    peak_rms_ratio=0.04,
)


class ReferenceOrigin(StrEnum):
    MINIMAL_VOICE_DESIGN = "minimal_voice_design"
    PREVIOUSLY_SUCCESSFUL_CONTROL = "previously_successful_control"


class AuditionCondition(StrEnum):
    COMPLETION = "completion"
    NON_FLOOR_FEEDBACK = "non_floor_feedback"
    INTERRUPTION_FLOOR_CLAIM = "interruption_floor_claim"


class VoiceReferenceCandidate(SyntheticModel):
    schema_version: Literal["voice-light-reference-ab-candidate-v1"] = (
        "voice-light-reference-ab-candidate-v1"
    )
    candidate_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    identity_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    candidate_index: int = Field(ge=1, le=2)
    identity_description: str = Field(min_length=1)
    origin: ReferenceOrigin
    reference_text: str = Field(min_length=1)
    voice_instruction: str = Field(min_length=1)
    request_seed: int = Field(ge=0)
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0)
    generation_seconds: float = Field(ge=0.0)


class VoiceReferenceAbManifest(SyntheticModel):
    schema_version: Literal["voice-light-reference-ab-manifest-v1"] = (
        "voice-light-reference-ab-manifest-v1"
    )
    backend: TtsBackendIdentity
    candidates: tuple[VoiceReferenceCandidate, ...] = Field(min_length=1)


class AuditionUtterance(SyntheticModel):
    utterance_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    condition: AuditionCondition
    text: str = Field(min_length=1)
    delivery_instruction: str = Field(min_length=1)
    assistant_gap_after_seconds: float = Field(ge=0.0, le=8.0)


class RenderedAuditionUtterance(SyntheticModel):
    candidate_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    utterance: AuditionUtterance
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: int = Field(gt=0)
    original_duration_seconds: float = Field(gt=0.0)
    duration_seconds: float = Field(gt=0.0)
    trimmed_leading_seconds: float = Field(ge=0.0)
    trimmed_trailing_seconds: float = Field(ge=0.0)
    continuation_silences: tuple[MeasuredSilence, ...] = ()
    generation_seconds: float = Field(ge=0.0)


class AuditionTimelineEntry(SyntheticModel):
    utterance_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    speech_start_seconds: float = Field(ge=0.0)
    speech_end_seconds: float = Field(gt=0.0)
    assistant_gap_end_seconds: float = Field(gt=0.0)


class RenderedCandidateConversation(SyntheticModel):
    candidate_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    audio_path: Path
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0)
    timeline: tuple[AuditionTimelineEntry, ...]


class VoiceReferenceAbRenderManifest(SyntheticModel):
    schema_version: Literal["voice-light-reference-ab-render-v1"] = (
        "voice-light-reference-ab-render-v1"
    )
    references_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: TtsBackendIdentity
    utterances: tuple[AuditionUtterance, ...]
    rendered_utterances: tuple[RenderedAuditionUtterance, ...]
    conversations: tuple[RenderedCandidateConversation, ...]


def minimal_voice_design_candidates() -> tuple[tuple[str, str, str, int], ...]:
    identities = (
        ("woman_easy", "Adult woman, easy natural conversational voice."),
        ("man_easy", "Adult man, easy natural conversational voice."),
        ("woman_lively", "Younger adult woman, lively everyday conversational voice."),
        ("man_relaxed", "Older adult man, relaxed everyday conversational voice."),
    )
    return tuple(
        (
            f"{identity_id}_{candidate_index}",
            identity_id,
            instruction,
            _candidate_seed(identity_id, candidate_index),
        )
        for identity_id, instruction in identities
        for candidate_index in (1, 2)
    )


def audition_utterances() -> tuple[AuditionUtterance, ...]:
    return (
        AuditionUtterance(
            utterance_id="opening",
            condition=AuditionCondition.COMPLETION,
            text=(
                "I finally moved the desk closer to the window, and the room feels completely "
                "different now."
            ),
            delivery_instruction="Brisk, natural conversation with a friend.",
            assistant_gap_after_seconds=2.0,
        ),
        AuditionUtterance(
            utterance_id="backchannel",
            condition=AuditionCondition.NON_FLOOR_FEEDBACK,
            text="Mm-hmm.",
            delivery_instruction="A very short, effortless listener acknowledgment.",
            assistant_gap_after_seconds=2.0,
        ),
        AuditionUtterance(
            utterance_id="follow_up",
            condition=AuditionCondition.COMPLETION,
            text=(
                "That makes sense. I hadn't realized how much the old layout was blocking the "
                "afternoon light."
            ),
            delivery_instruction="Brisk, engaged, natural conversation.",
            assistant_gap_after_seconds=2.5,
        ),
        AuditionUtterance(
            utterance_id="interruption",
            condition=AuditionCondition.INTERRUPTION_FLOOR_CLAIM,
            text="Wait, does that mean the delivery is delayed again?",
            delivery_instruction="A quick, spontaneous interruption.",
            assistant_gap_after_seconds=1.5,
        ),
        AuditionUtterance(
            utterance_id="closing",
            condition=AuditionCondition.COMPLETION,
            text=(
                "I can work around Friday if I know by tomorrow, but after that I'll need to "
                "rearrange the whole weekend, so I'd rather confirm the schedule before making "
                "any other plans."
            ),
            delivery_instruction="Natural, engaged conversation with lightly varied pacing.",
            assistant_gap_after_seconds=0.0,
        ),
    )


def copy_control_reference(
    source_path: Path,
    destination_path: Path,
    candidate_id: str,
    identity_id: str,
    identity_description: str,
    reference_text: str,
    voice_instruction: str,
    request_seed: int,
) -> VoiceReferenceCandidate:
    sample_rate_hz, samples = read_pcm16_wave(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return VoiceReferenceCandidate(
        candidate_id=candidate_id,
        identity_id=identity_id,
        candidate_index=1,
        identity_description=identity_description,
        origin=ReferenceOrigin.PREVIOUSLY_SUCCESSFUL_CONTROL,
        reference_text=reference_text,
        voice_instruction=voice_instruction,
        request_seed=request_seed,
        audio_path=destination_path,
        audio_sha256=file_sha256(destination_path),
        sample_rate_hz=sample_rate_hz,
        duration_seconds=samples.size / sample_rate_hz,
        generation_seconds=0.0,
    )


def compose_candidate_conversation(
    candidate_id: str,
    rendered_utterances: tuple[RenderedAuditionUtterance, ...],
    output_path: Path,
) -> RenderedCandidateConversation:
    if not rendered_utterances:
        raise ValueError(f"Candidate {candidate_id} has no rendered audition utterances.")
    sample_rates = {utterance.sample_rate_hz for utterance in rendered_utterances}
    if len(sample_rates) != 1:
        raise ValueError(f"Candidate {candidate_id} has inconsistent sample rates.")
    sample_rate_hz = sample_rates.pop()
    chunks: list[np.ndarray] = []
    timeline: list[AuditionTimelineEntry] = []
    cursor_samples = 0
    for rendered in rendered_utterances:
        audio_sample_rate_hz, samples = read_pcm16_wave(rendered.audio_path)
        if audio_sample_rate_hz != sample_rate_hz:
            raise ValueError(f"Candidate {candidate_id} WAV sample rate changed on disk.")
        speech_start_seconds = cursor_samples / sample_rate_hz
        chunks.append(samples)
        cursor_samples += samples.size
        speech_end_seconds = cursor_samples / sample_rate_hz
        gap_samples = round(rendered.utterance.assistant_gap_after_seconds * sample_rate_hz)
        if gap_samples:
            chunks.append(np.zeros(gap_samples, dtype=np.float32))
            cursor_samples += gap_samples
        timeline.append(
            AuditionTimelineEntry(
                utterance_id=rendered.utterance.utterance_id,
                speech_start_seconds=speech_start_seconds,
                speech_end_seconds=speech_end_seconds,
                assistant_gap_end_seconds=cursor_samples / sample_rate_hz,
            )
        )
    conversation = np.concatenate(chunks)
    write_pcm16_wave(output_path, conversation, sample_rate_hz)
    return RenderedCandidateConversation(
        candidate_id=candidate_id,
        audio_path=output_path,
        audio_sha256=file_sha256(output_path),
        sample_rate_hz=sample_rate_hz,
        duration_seconds=conversation.size / sample_rate_hz,
        timeline=tuple(timeline),
    )


def materialize_trimmed_audition_utterance(
    candidate_id: str,
    utterance: AuditionUtterance,
    samples: np.ndarray,
    sample_rate_hz: int,
    generation_seconds: float,
    output_path: Path,
    detection: SilenceDetectionConfiguration = AUDITION_SILENCE_DETECTION,
) -> RenderedAuditionUtterance:
    item_id = f"{candidate_id}/{utterance.utterance_id}"
    trimmed = trim_generated_speech(samples, sample_rate_hz, item_id, detection)
    continuation_silences = measure_internal_silences(
        trimmed.active_frames,
        trimmed.frame_seconds,
        detection.minimum_silence_milliseconds / 1000.0,
        trimmed.samples.size / sample_rate_hz,
    )
    write_pcm16_wave(output_path, trimmed.samples, sample_rate_hz)
    return RenderedAuditionUtterance(
        candidate_id=candidate_id,
        utterance=utterance,
        audio_path=output_path,
        audio_sha256=file_sha256(output_path),
        sample_rate_hz=sample_rate_hz,
        original_duration_seconds=trimmed.original_duration_seconds,
        duration_seconds=trimmed.samples.size / sample_rate_hz,
        trimmed_leading_seconds=trimmed.leading_seconds,
        trimmed_trailing_seconds=trimmed.trailing_seconds,
        continuation_silences=continuation_silences,
        generation_seconds=generation_seconds,
    )


def render_voice_reference_ab_review(
    references_path: Path,
    renders_path: Path,
    output_path: Path,
) -> Path:
    references = load_reference_manifest(references_path)
    renders = VoiceReferenceAbRenderManifest.model_validate_json(
        renders_path.read_text(encoding="utf-8")
    )
    if renders.references_sha256 != file_sha256(references_path):
        raise ValueError("CosyVoice audition manifest belongs to different references.")
    reference_directory = references_path.parent
    render_directory = renders_path.parent
    review_directory = output_path.parent
    units_by_candidate = {
        candidate.candidate_id: tuple(
            rendered
            for rendered in renders.rendered_utterances
            if rendered.candidate_id == candidate.candidate_id
        )
        for candidate in references.candidates
    }
    conversations_by_candidate = {
        conversation.candidate_id: conversation for conversation in renders.conversations
    }
    cards = "".join(
        _review_card(
            candidate,
            units_by_candidate[candidate.candidate_id],
            conversations_by_candidate[candidate.candidate_id],
            reference_directory,
            render_directory,
            review_directory,
        )
        for candidate in references.candidates
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_review_document(cards), encoding="utf-8")
    return output_path


def load_reference_manifest(path: Path) -> VoiceReferenceAbManifest:
    return VoiceReferenceAbManifest.model_validate_json(path.read_text(encoding="utf-8"))


def read_pcm16_wave(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as audio_file:
        if audio_file.getnchannels() != 1 or audio_file.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16 WAV: {path}")
        sample_rate_hz = audio_file.getframerate()
        samples = np.frombuffer(audio_file.readframes(audio_file.getnframes()), dtype="<i2").astype(
            np.float32
        )
    return sample_rate_hz, samples / 32767.0


def _candidate_seed(identity_id: str, candidate_index: int) -> int:
    content = f"voice-reference-ab-v1:{identity_id}:{candidate_index}"
    return int.from_bytes(hashlib.sha256(content.encode()).digest()[:4], "big")


def _review_card(
    candidate: VoiceReferenceCandidate,
    units: tuple[RenderedAuditionUtterance, ...],
    conversation: RenderedCandidateConversation,
    reference_directory: Path,
    render_directory: Path,
    review_directory: Path,
) -> str:
    reference_url = _relative_url(reference_directory / candidate.audio_path, review_directory)
    conversation_url = _relative_url(render_directory / conversation.audio_path, review_directory)
    unit_fragments = []
    for unit in units:
        unit_url = _relative_url(render_directory / unit.audio_path, review_directory)
        unit_fragments.append(
            "".join(
                (
                    "<article class='unit'>",
                    f"<span>{html.escape(unit.utterance.condition.value)}</span>",
                    f"<p>{html.escape(unit.utterance.text)}</p>",
                    f"<audio controls preload='none' src='{html.escape(unit_url)}'></audio>",
                    "</article>",
                )
            )
        )
    unit_html = "".join(unit_fragments)
    heading = f"{candidate.identity_id} · candidate {candidate.candidate_index}"
    return "".join(
        (
            "<section class='candidate'>",
            f"<h2>{html.escape(heading)}</h2>",
            f"<p class='origin'>{html.escape(candidate.origin.value)}</p>",
            f"<p><strong>Identity:</strong> {html.escape(candidate.identity_description)}</p>",
            f"<p><strong>Qwen instruction:</strong> {html.escape(candidate.voice_instruction)}</p>",
            "<p><strong>Reference transcript:</strong> ",
            f"{html.escape(candidate.reference_text)}</p>",
            f"<audio controls preload='none' src='{html.escape(reference_url)}'></audio>",
            "<h3>Identical complete CosyVoice conversation</h3>",
            f"<audio controls preload='none' src='{html.escape(conversation_url)}'></audio>",
            "<details><summary>Isolated cloned segments</summary>",
            unit_html,
            "</details></section>",
        )
    )


def _relative_url(path: Path, review_directory: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Review audio is missing: {path}")
    return Path(os.path.relpath(path.resolve(), review_directory.resolve())).as_posix()


def _review_document(cards: str) -> str:
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Qwen reference → CosyVoice A/B</title><style>
body{{font:16px system-ui;background:#10141d;color:#edf2fa;max-width:1050px;
margin:auto;padding:24px}}
.candidate{{background:#192131;border:1px solid #34435b;border-radius:14px;
padding:18px;margin:18px 0}}
.origin,.unit span{{color:#9fc6ff}} audio{{width:100%;margin:8px 0 14px}} details{{margin-top:12px}}
.unit{{border-top:1px solid #34435b;padding:10px 0}} h1,h2,h3{{line-height:1.2}}
</style></head><body><h1>Qwen VoiceDesign → CosyVoice reference A/B</h1>
<p>All candidates use the same reference transcript and identical cloned conversation.
Two previously successful references are included as controls. Nothing is filtered.</p>
{cards}</body></html>"""
