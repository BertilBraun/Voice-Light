from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from app.local.synthetic_generation.provider_listening_pilot import (
    load_provider_listening_manifest,
    render_listening_review,
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    manifests = tuple(load_provider_listening_manifest(path) for path in parsed.manifests)
    render_listening_review(manifests, parsed.output)
    print(parsed.output, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the TTS provider listening review page.")
    parser.add_argument("--manifest", dest="manifests", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
