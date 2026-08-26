from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.local.synthetic_generation.conversation_compiler import ConversationCompilerConfig
from app.local.synthetic_generation.conversation_pipeline import build_conversation_corpus
from app.local.synthetic_generation.corpus import (
    SyntheticCorpusRequest,
    build_synthetic_corpus,
)
from app.local.synthetic_generation.models import SyntheticConversationPlan


def main(arguments: Sequence[str] | None = None) -> None:
    parser = _parser()
    parsed = parser.parse_args(arguments)
    try:
        match parsed.command:
            case "validate-plan":
                plan = SyntheticConversationPlan.model_validate_json(
                    parsed.plan.read_text(encoding="utf-8")
                )
                print(plan.model_dump_json(indent=2), flush=True)
            case "build-corpus":
                request = SyntheticCorpusRequest.model_validate_json(
                    parsed.request.read_text(encoding="utf-8")
                )
                manifest = build_synthetic_corpus(request, parsed.output)
                print(manifest.model_dump_json(indent=2), flush=True)
            case "compile-conversations":
                manifest = build_conversation_corpus(
                    prompt_set_path=parsed.prompt_set,
                    tts_manifest_path=parsed.tts_manifest,
                    output_directory=parsed.output,
                    split_seed=parsed.split_seed,
                    compiler_config=ConversationCompilerConfig(
                        crop_variant_count=parsed.crop_variants,
                        event_light_fraction=parsed.event_light_fraction,
                        assistant_duration_variation=parsed.assistant_duration_variation,
                    ),
                )
                print(manifest.model_dump_json(indent=2), flush=True)
            case _:
                raise AssertionError(f"Unexpected command: {parsed.command}")
    except (OSError, ValidationError, ValueError) as error:
        parser.error(str(error))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and build planned two-party synthetic training corpora."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-plan")
    validate_parser.add_argument("plan", type=Path)
    build_parser = subparsers.add_parser("build-corpus")
    build_parser.add_argument("--request", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    conversation_parser = subparsers.add_parser(
        "compile-conversations",
        help="Compile rendered user-only conversations into training shards.",
    )
    conversation_parser.add_argument("--prompt-set", required=True, type=Path)
    conversation_parser.add_argument("--tts-manifest", required=True, type=Path)
    conversation_parser.add_argument("--output", required=True, type=Path)
    conversation_parser.add_argument("--split-seed", default="synthetic-conversation-pilot-v1")
    conversation_parser.add_argument("--crop-variants", default=4, type=int)
    conversation_parser.add_argument("--event-light-fraction", default=0.15, type=float)
    conversation_parser.add_argument("--assistant-duration-variation", default=0.1, type=float)
    return parser


if __name__ == "__main__":
    main()
