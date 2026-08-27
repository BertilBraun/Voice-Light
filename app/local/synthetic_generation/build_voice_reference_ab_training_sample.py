from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from app.local.synthetic_generation.conversation_compiler import (
    ConversationCompilerConfig,
    materialize_crop_audio,
)
from app.local.synthetic_generation.voice_reference_ab_pilot import (
    VoiceReferenceAbRenderManifest,
)
from app.local.synthetic_generation.voice_reference_ab_training import (
    compile_audition_candidate_training_samples,
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    render_manifest = VoiceReferenceAbRenderManifest.model_validate_json(
        parsed.render_manifest.read_text(encoding="utf-8")
    )
    render_directory = parsed.render_manifest.parent
    rendered = tuple(
        utterance.model_copy(update={"audio_path": render_directory / utterance.audio_path})
        for utterance in render_manifest.rendered_utterances
    )
    compiled = compile_audition_candidate_training_samples(
        candidate_id=parsed.candidate_id,
        rendered_utterances=rendered,
        output_directory=parsed.output,
        config=ConversationCompilerConfig(
            crop_variant_count=parsed.crop_count,
            assistant_only_fraction=0.0,
            user_only_fraction=0.0,
            event_light_fraction=0.0,
        ),
    )
    crop_directory = parsed.output / "crops"
    for crop in compiled.crops:
        materialize_crop_audio(
            compiled,
            crop,
            crop_directory / f"variant-{crop.variant_index}.wav",
        )
    manifest_path = parsed.output / "compiled.json"
    manifest_path.write_text(compiled.model_dump_json(indent=2), encoding="utf-8")
    print(manifest_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile one trimmed A/B candidate into 20-second training samples."
    )
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crop-count", default=4, type=int)
    return parser


if __name__ == "__main__":
    main()
