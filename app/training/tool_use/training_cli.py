from __future__ import annotations

import argparse
from pathlib import Path

from app.training.tool_use.training import run_lora_training
from app.training.tool_use.training_config import pilot_training_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the approved Voice Light Qwen3-1.7B BF16 LoRA pilot."
    )
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Render and validate the corpus without loading the base model.",
    )
    arguments = parser.parse_args()
    config = pilot_training_config(
        records_path=arguments.records,
        output_directory=arguments.output,
        source_commit=arguments.source_commit,
    )
    summary = run_lora_training(config=config, prepare_only=arguments.prepare_only)
    if summary is None:
        print(f"Prepared and validated the training corpus at {arguments.output}")
    else:
        print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
