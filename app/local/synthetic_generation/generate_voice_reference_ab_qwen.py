from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

from app.local.synthetic_generation.conversation_voice_references import (
    ReferenceSynthesisRequest,
    ReferenceSynthesisResult,
    file_sha256,
    trim_generated_speech,
    write_pcm16_wave,
)
from app.local.synthetic_generation.generate_conversation_qwen_references import (
    QwenVoiceDesignReferenceSynthesizer,
)
from app.local.synthetic_generation.voice_reference_ab_pilot import (
    REFERENCE_TEXT,
    ReferenceOrigin,
    VoiceReferenceAbManifest,
    VoiceReferenceCandidate,
    copy_control_reference,
    minimal_voice_design_candidates,
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    model = Qwen3TTSModel.from_pretrained(
        parsed.model,
        revision=parsed.model_revision,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    synthesizer = QwenVoiceDesignReferenceSynthesizer(
        model=model,
        model_id=parsed.model,
        model_revision=parsed.model_revision,
    )
    output_directory: Path = parsed.output
    audio_directory = output_directory / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)
    candidates = list(_generate_candidates(synthesizer, audio_directory, parsed.batch_size))
    candidates.extend(
        _control_candidates(
            parsed.control_one,
            parsed.control_two,
            audio_directory,
        )
    )
    relative_candidates = tuple(
        candidate.model_copy(
            update={"audio_path": candidate.audio_path.relative_to(output_directory)}
        )
        for candidate in candidates
    )
    manifest = VoiceReferenceAbManifest(
        backend=synthesizer.identity,
        candidates=relative_candidates,
    )
    manifest_path = output_directory / "references.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    print(manifest_path, flush=True)


def _generate_candidates(
    synthesizer: QwenVoiceDesignReferenceSynthesizer,
    audio_directory: Path,
    batch_size: int,
) -> tuple[VoiceReferenceCandidate, ...]:
    if batch_size <= 0:
        raise ValueError("Qwen reference A/B batch size must be positive.")
    specifications = minimal_voice_design_candidates()
    requests = tuple(
        ReferenceSynthesisRequest(
            plan_id=candidate_id,
            text=REFERENCE_TEXT,
            voice_instruction=voice_instruction,
            seed=seed,
        )
        for candidate_id, _, voice_instruction, seed in specifications
    )
    results_by_id: dict[str, ReferenceSynthesisResult] = {}
    for offset in range(0, len(requests), batch_size):
        batch = requests[offset : offset + batch_size]
        for result in synthesizer.generate_batch(batch):
            results_by_id[result.plan_id] = result
        print(f"qwen_references={len(results_by_id)}/{len(requests)}", flush=True)
    candidates = []
    for candidate_id, identity_id, voice_instruction, seed in specifications:
        result = results_by_id[candidate_id]
        trimmed = trim_generated_speech(
            result.samples,
            result.sample_rate_hz,
            candidate_id,
        )
        audio_path = audio_directory / f"{candidate_id}.wav"
        write_pcm16_wave(audio_path, trimmed.samples, trimmed.sample_rate_hz)
        duration_seconds = trimmed.samples.size / trimmed.sample_rate_hz
        candidates.append(
            VoiceReferenceCandidate(
                candidate_id=candidate_id,
                identity_id=identity_id,
                candidate_index=int(candidate_id.rsplit("_", maxsplit=1)[1]),
                identity_description=voice_instruction,
                origin=ReferenceOrigin.MINIMAL_VOICE_DESIGN,
                reference_text=REFERENCE_TEXT,
                voice_instruction=voice_instruction,
                request_seed=seed,
                audio_path=audio_path,
                audio_sha256=file_sha256(audio_path),
                sample_rate_hz=trimmed.sample_rate_hz,
                duration_seconds=duration_seconds,
                generation_seconds=result.generation_seconds,
            )
        )
    return tuple(candidates)


def _control_candidates(
    control_one_path: Path,
    control_two_path: Path,
    audio_directory: Path,
) -> tuple[VoiceReferenceCandidate, ...]:
    return (
        copy_control_reference(
            source_path=control_one_path,
            destination_path=audio_directory / "control_previous_one.wav",
            candidate_id="control_previous_one",
            identity_id="control_previous_one",
            identity_description="Previously successful general-American Qwen reference.",
            reference_text=(
                "A calm voice explains a simple idea with clear and steady pacing, spoken "
                "clearly and steadily for this short recording."
            ),
            voice_instruction=(
                "Adult general-American speaker, medium pitch and light vocal weight; moderate "
                "pace, conversational energy, neutral affect."
            ),
            request_seed=3525283668,
        ),
        copy_control_reference(
            source_path=control_two_path,
            destination_path=audio_directory / "control_previous_two.wav",
            candidate_id="control_previous_two",
            identity_id="control_previous_two",
            identity_description="Previously successful southern-British Qwen reference.",
            reference_text=(
                "A person explains a simple idea clearly and calmly, spoken clearly and steadily "
                "for this short recording."
            ),
            voice_instruction=(
                "Adult southern-British speaker, high pitch and full vocal weight; moderate pace, "
                "conversational energy, neutral affect."
            ),
            request_seed=1897734633,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate minimally prompted Qwen voice references and preserve controls."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-one", required=True, type=Path)
    parser.add_argument("--control-two", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--batch-size", default=2, type=int)
    return parser


if __name__ == "__main__":
    main()
