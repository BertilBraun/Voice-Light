from __future__ import annotations

import argparse
from pathlib import Path

from app.training.turn_taking.evaluation import EvaluationReport, compare_evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two turn-taking evaluations.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()
    reference = EvaluationReport.model_validate_json(
        arguments.reference.read_text(encoding="utf-8")
    )
    candidate = EvaluationReport.model_validate_json(
        arguments.candidate.read_text(encoding="utf-8")
    )
    comparison = compare_evaluations(reference, candidate)
    print(comparison.model_dump_json(), flush=True)
    if not comparison.improved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
