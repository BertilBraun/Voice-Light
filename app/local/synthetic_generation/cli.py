from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

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
    return parser


if __name__ == "__main__":
    main()
