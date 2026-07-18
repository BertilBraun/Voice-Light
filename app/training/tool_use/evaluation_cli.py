from __future__ import annotations

import argparse
from pathlib import Path

from app.training.tool_use.evaluation import run_generation_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base and LoRA tool behavior on the balanced holdout."
    )
    parser.add_argument("records", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    arguments = parser.parse_args()
    report = run_generation_evaluation(
        records_path=arguments.records,
        adapter_path=arguments.adapter,
        output_directory=arguments.output,
        random_seeds=(17, 29, 43),
        batch_size=arguments.batch_size,
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
