from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from app.local.synthetic_generation.voice_reference_ab_pilot import (
    render_voice_reference_ab_review,
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    output_path = render_voice_reference_ab_review(
        references_path=parsed.references,
        renders_path=parsed.renders,
        output_path=parsed.output,
    )
    print(output_path, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Qwen reference versus CosyVoice listening A/B page."
    )
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
