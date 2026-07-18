from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from app.training.tool_use.generation import RolloutGenerationConfig, generate_dataset
from app.training.tool_use.scenario import read_scenarios, sample_scenarios, write_scenarios
from app.training.tool_use.schema import read_records, validate_records
from app.training.tool_use.statistics import dataset_statistics
from app.training.tool_use.vllm_client import VllmClientConfig, VllmStructuredClient


def main() -> None:
    parser = _argument_parser()
    arguments = parser.parse_args()
    match arguments.command:
        case "plan":
            _plan(arguments)
        case "generate":
            asyncio.run(_generate(arguments))
        case "validate":
            _validate(arguments)
        case "summarize":
            _summarize(arguments)
        case _:
            raise AssertionError("argparse returned an excluded tool-use command.")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare synthetic Voice Light tool-use data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Create deterministic scenario JSONL.")
    plan_parser.add_argument("output", type=Path)
    plan_parser.add_argument("--count", type=int, required=True)
    plan_parser.add_argument("--seed", type=int, required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run resumable staged rollouts against a vLLM server.",
    )
    generate_parser.add_argument("scenarios", type=Path)
    generate_parser.add_argument("output", type=Path)
    generate_parser.add_argument("--failures", type=Path, required=True)
    generate_parser.add_argument("--manifest", type=Path, required=True)
    generate_parser.add_argument("--request-log", type=Path, required=True)
    generate_parser.add_argument("--base-url", required=True)
    generate_parser.add_argument("--api-key", required=True)
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--model-revision", required=True)
    generate_parser.add_argument("--quantization", required=True)
    generate_parser.add_argument(
        "--time-reference",
        type=_timezone_aware_datetime,
        required=True,
        help="Fixed ISO 8601 anchor used to sample reproducible times from the prior year.",
    )
    generate_parser.add_argument("--limit", type=int)
    generate_parser.add_argument("--concurrency", type=int, default=128)
    generate_parser.add_argument("--semantic-attempts", type=int, default=3)
    generate_parser.add_argument("--http-attempts", type=int, default=3)
    generate_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-p", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=20)
    generate_parser.add_argument("--min-p", type=float, default=0.0)
    generate_parser.add_argument("--presence-penalty", type=float, default=1.5)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    generate_parser.add_argument("--maximum-tokens", type=int, default=1200)

    validate_parser = subparsers.add_parser("validate", help="Validate canonical record JSONL.")
    validate_parser.add_argument("records", type=Path)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Print deterministic corpus statistics.",
    )
    summarize_parser.add_argument("records", type=Path)
    return parser


def _plan(arguments: argparse.Namespace) -> None:
    scenarios = sample_scenarios(count=arguments.count, random_seed=arguments.seed)
    write_scenarios(arguments.output, scenarios)
    print(f"Wrote {len(scenarios)} scenarios to {arguments.output}")


async def _generate(arguments: argparse.Namespace) -> None:
    scenarios = read_scenarios(arguments.scenarios)
    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError("--limit must be positive.")
        scenarios = scenarios[: arguments.limit]
    client_config = VllmClientConfig(
        base_url=arguments.base_url,
        api_key=arguments.api_key,
        model_identifier=arguments.model,
        request_timeout_seconds=arguments.timeout_seconds,
        maximum_http_attempts=arguments.http_attempts,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        top_k=arguments.top_k,
        min_p=arguments.min_p,
        presence_penalty=arguments.presence_penalty,
        repetition_penalty=arguments.repetition_penalty,
        maximum_tokens=arguments.maximum_tokens,
    )
    generation_config = RolloutGenerationConfig(
        model_identifier=arguments.model,
        model_revision=arguments.model_revision,
        quantization=arguments.quantization,
        time_reference=arguments.time_reference,
        maximum_concurrency=arguments.concurrency,
        maximum_semantic_attempts=arguments.semantic_attempts,
    )
    async with VllmStructuredClient(
        config=client_config,
        request_log_path=arguments.request_log,
    ) as client:
        result = await generate_dataset(
            scenarios=scenarios,
            generator=client,
            config=generation_config,
            output_path=arguments.output,
            failure_path=arguments.failures,
            manifest_path=arguments.manifest,
        )
    print(result.manifest.model_dump_json(indent=2))


def _validate(arguments: argparse.Namespace) -> None:
    records = read_records(arguments.records)
    validate_records(records)
    print(f"Validated {len(records)} canonical records.")


def _summarize(arguments: argparse.Namespace) -> None:
    records = read_records(arguments.records)
    print(dataset_statistics(records).model_dump_json(indent=2))


def _timezone_aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected an ISO 8601 timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("Timestamp must include a UTC offset.")
    return parsed


if __name__ == "__main__":
    main()
