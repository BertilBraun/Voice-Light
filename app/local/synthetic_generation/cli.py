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
from app.local.synthetic_generation.synthetic_hub import (
    DEFAULT_SYNTHETIC_HUB_REPOSITORY,
    SyntheticHubPreparationRequest,
    prepare_synthetic_hub_corpora,
)
from app.local.synthetic_generation.synthetic_publication import (
    SyntheticPublicationRequest,
    stage_synthetic_publication,
    summarize_synthetic_publication,
)


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
                        assistant_only_fraction=parsed.assistant_only_fraction,
                        user_only_fraction=parsed.user_only_fraction,
                        event_light_fraction=parsed.event_light_fraction,
                        assistant_duration_variation=parsed.assistant_duration_variation,
                    ),
                    enforce_sampling_gates=not parsed.allow_incomplete_sampling_controls,
                )
                print(manifest.model_dump_json(indent=2), flush=True)
            case "prepare-hub-training":
                manifest = prepare_synthetic_hub_corpora(
                    SyntheticHubPreparationRequest(
                        repository_id=parsed.repository,
                        revision=parsed.revision,
                        run_ids=tuple(parsed.run),
                        cache_directory=parsed.cache_directory,
                        output_directory=parsed.output,
                        split_seed=parsed.split_seed,
                        compiler=ConversationCompilerConfig(
                            crop_variant_count=parsed.crop_variants,
                            assistant_only_fraction=parsed.assistant_only_fraction,
                            user_only_fraction=parsed.user_only_fraction,
                            event_light_fraction=parsed.event_light_fraction,
                            assistant_duration_variation=parsed.assistant_duration_variation,
                        ),
                    )
                )
                print(manifest.model_dump_json(indent=2), flush=True)
            case "stage-public-run":
                destination = parsed.staging_root / "runs" / parsed.run_id
                manifest = stage_synthetic_publication(
                    SyntheticPublicationRequest(
                        run_id=parsed.run_id,
                        source_directory=parsed.source,
                        staging_root=parsed.staging_root,
                        source_code_revision=parsed.source_code_revision,
                    )
                )
                print(
                    summarize_synthetic_publication(manifest, destination).model_dump_json(
                        indent=2
                    ),
                    flush=True,
                )
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
    conversation_parser.add_argument("--crop-variants", default=8, type=int)
    conversation_parser.add_argument("--assistant-only-fraction", default=0.1, type=float)
    conversation_parser.add_argument("--user-only-fraction", default=0.1, type=float)
    conversation_parser.add_argument("--event-light-fraction", default=0.1, type=float)
    conversation_parser.add_argument("--assistant-duration-variation", default=0.1, type=float)
    conversation_parser.add_argument(
        "--allow-incomplete-sampling-controls",
        action="store_true",
        help="Allow small review pilots that cannot meet corpus-scale control quotas.",
    )
    hub_parser = subparsers.add_parser(
        "prepare-hub-training",
        help="Download source speech units and materialize local training corpora.",
    )
    hub_parser.add_argument("--repository", default=DEFAULT_SYNTHETIC_HUB_REPOSITORY)
    hub_parser.add_argument("--revision", required=True)
    hub_parser.add_argument("--run", required=True, action="append")
    hub_parser.add_argument("--cache-directory", type=Path)
    hub_parser.add_argument("--output", required=True, type=Path)
    hub_parser.add_argument("--split-seed", default="voice-light-synthetic-training-v1")
    hub_parser.add_argument("--crop-variants", default=4, type=int)
    hub_parser.add_argument("--assistant-only-fraction", default=0.1, type=float)
    hub_parser.add_argument("--user-only-fraction", default=0.1, type=float)
    hub_parser.add_argument("--event-light-fraction", default=0.1, type=float)
    hub_parser.add_argument("--assistant-duration-variation", default=0.1, type=float)
    publication_parser = subparsers.add_parser(
        "stage-public-run",
        help="Copy a completed corpus into the portable public dataset layout.",
    )
    publication_parser.add_argument("--run-id", required=True)
    publication_parser.add_argument("--source", required=True, type=Path)
    publication_parser.add_argument("--staging-root", required=True, type=Path)
    publication_parser.add_argument("--source-code-revision", required=True)
    return parser


if __name__ == "__main__":
    main()
