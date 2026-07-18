from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

import torch
from peft import PeftModel
from pydantic import Field, ValidationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.compute.voice.hermes_tool_parser import (
    HermesSpokenText,
    HermesToolCallCompleted,
    HermesToolCallFailed,
    HermesToolCallParser,
)
from app.compute.voice.tools import (
    CalculateArguments,
    GetTimeArguments,
    SearchArguments,
    SerializedToolCall,
    ToolName,
)
from app.training.tool_use.renderer import (
    QwenChatMessage,
    QwenToolSpecification,
    qwen_message,
    qwen_tool_definition,
)
from app.training.tool_use.schema import (
    AssistantMessage,
    DatasetSplit,
    ToolDialogueRecord,
    ToolResultMessage,
    ToolUseBaseModel,
    read_records,
)
from app.training.tool_use.training_config import (
    QWEN3_1_7B_MODEL_IDENTIFIER,
    QWEN3_1_7B_REVISION,
)
from app.training.tool_use.training_data import assign_proof_of_concept_splits


class EvaluationCaseKind(StrEnum):
    TOOL_CALL = "tool_call"
    TOOL_CONTINUATION = "tool_continuation"
    NO_TOOL = "no_tool"


class ModelVariant(StrEnum):
    BASE = "base"
    ADAPTER = "adapter"


@dataclass(frozen=True)
class HeldoutGenerationCase:
    case_id: str
    record_id: str
    kind: EvaluationCaseKind
    tools: tuple[QwenToolSpecification, ...]
    prefix_messages: tuple[QwenChatMessage, ...]
    expected_tool_name: str | None


class GenerationPrediction(ToolUseBaseModel):
    schema_version: Literal["voice-light.generation-prediction/v1"] = (
        "voice-light.generation-prediction/v1"
    )
    case_id: str
    record_id: str
    case_kind: EvaluationCaseKind
    model_variant: ModelVariant
    random_seed: int
    generated_text: str
    spoken_text: str
    expected_tool_name: str | None
    generated_tool_name: str | None
    parser_valid: bool
    tool_schema_valid: bool
    bridge_word_count: int = Field(ge=0)


class GenerationMetrics(ToolUseBaseModel):
    model_variant: ModelVariant
    prediction_count: int = Field(ge=1)
    parser_valid_rate: float = Field(ge=0.0, le=1.0)
    tool_decision_accuracy: float = Field(ge=0.0, le=1.0)
    tool_name_accuracy: float = Field(ge=0.0, le=1.0)
    tool_schema_valid_rate: float = Field(ge=0.0, le=1.0)
    bridge_present_rate: float = Field(ge=0.0, le=1.0)
    bridge_concise_rate: float = Field(ge=0.0, le=1.0)
    no_tool_accuracy: float = Field(ge=0.0, le=1.0)
    continuation_nonempty_rate: float = Field(ge=0.0, le=1.0)


class GenerationEvaluationReport(ToolUseBaseModel):
    schema_version: Literal["voice-light.generation-evaluation/v1"] = (
        "voice-light.generation-evaluation/v1"
    )
    model_identifier: str
    model_revision: str
    adapter_path: Path
    holdout_record_count: int = Field(ge=1)
    case_count: int = Field(ge=1)
    random_seeds: tuple[int, ...] = Field(min_length=1)
    maximum_new_tokens: int = Field(ge=1)
    temperature: float = Field(gt=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    base: GenerationMetrics
    adapter: GenerationMetrics


def run_generation_evaluation(
    records_path: Path,
    adapter_path: Path,
    output_directory: Path,
    random_seeds: tuple[int, ...],
    batch_size: int,
) -> GenerationEvaluationReport:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(f"Evaluation output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    source_records = read_records(records_path)
    assigned_records = assign_proof_of_concept_splits(
        source_records=source_records,
        holdout_record_count=80,
        random_seed=20260729,
    )
    cases = heldout_generation_cases(assigned_records)
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN3_1_7B_MODEL_IDENTIFIER,
        revision=QWEN3_1_7B_REVISION,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        raise ValueError("The target tokenizer has no padding token.")
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN3_1_7B_MODEL_IDENTIFIER,
        revision=QWEN3_1_7B_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    base_predictions = _generate_predictions(
        model=base_model,
        tokenizer=tokenizer,
        cases=cases,
        model_variant=ModelVariant.BASE,
        random_seeds=random_seeds,
        batch_size=batch_size,
    )
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapter_predictions = _generate_predictions(
        model=adapter_model,
        tokenizer=tokenizer,
        cases=cases,
        model_variant=ModelVariant.ADAPTER,
        random_seeds=random_seeds,
        batch_size=batch_size,
    )
    predictions = (*base_predictions, *adapter_predictions)
    _write_predictions(output_directory / "predictions.jsonl", predictions)
    report = GenerationEvaluationReport(
        model_identifier=QWEN3_1_7B_MODEL_IDENTIFIER,
        model_revision=QWEN3_1_7B_REVISION,
        adapter_path=adapter_path,
        holdout_record_count=80,
        case_count=len(cases),
        random_seeds=random_seeds,
        maximum_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        base=generation_metrics(base_predictions),
        adapter=generation_metrics(adapter_predictions),
    )
    (output_directory / "report.json").write_text(
        f"{report.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    return report


def heldout_generation_cases(
    assigned_records: tuple[ToolDialogueRecord, ...],
) -> tuple[HeldoutGenerationCase, ...]:
    validation_records = tuple(
        record
        for record in assigned_records
        if record.metadata.split.name is DatasetSplit.VALIDATION
    )
    cases: list[HeldoutGenerationCase] = []
    for record in validation_records:
        record_has_tool_call = any(
            message.tool_calls
            for message in record.messages
            if isinstance(message, AssistantMessage)
        )
        no_tool_case_added = False
        for message_index, message in enumerate(record.messages):
            match message:
                case AssistantMessage(tool_calls=tool_calls):
                    previous_is_tool_result = message_index > 0 and isinstance(
                        record.messages[message_index - 1], ToolResultMessage
                    )
                    if tool_calls:
                        kind = EvaluationCaseKind.TOOL_CALL
                        expected_tool_name = tool_calls[0].tool_name
                    elif previous_is_tool_result:
                        kind = EvaluationCaseKind.TOOL_CONTINUATION
                        expected_tool_name = None
                    elif not record_has_tool_call and not no_tool_case_added:
                        kind = EvaluationCaseKind.NO_TOOL
                        expected_tool_name = None
                        no_tool_case_added = True
                    else:
                        continue
                    cases.append(
                        HeldoutGenerationCase(
                            case_id=f"{record.record_id}-assistant-{message_index}",
                            record_id=record.record_id,
                            kind=kind,
                            tools=tuple(qwen_tool_definition(tool) for tool in record.tools),
                            prefix_messages=tuple(
                                qwen_message(prefix_message)
                                for prefix_message in record.messages[:message_index]
                            ),
                            expected_tool_name=expected_tool_name,
                        )
                    )
                case _:
                    continue
    return tuple(cases)


def parse_generation_prediction(
    case: HeldoutGenerationCase,
    model_variant: ModelVariant,
    random_seed: int,
    generated_text: str,
) -> GenerationPrediction:
    parser = HermesToolCallParser(invocation_id=1)
    events = (*parser.add_text(generated_text), *parser.finish())
    spoken_text = "".join(
        event.text for event in events if isinstance(event, HermesSpokenText)
    ).strip()
    calls = tuple(event.request for event in events if isinstance(event, HermesToolCallCompleted))
    failures = tuple(event for event in events if isinstance(event, HermesToolCallFailed))
    generated_call = calls[0] if len(calls) == 1 else None
    return GenerationPrediction(
        case_id=case.case_id,
        record_id=case.record_id,
        case_kind=case.kind,
        model_variant=model_variant,
        random_seed=random_seed,
        generated_text=generated_text,
        spoken_text=spoken_text,
        expected_tool_name=case.expected_tool_name,
        generated_tool_name=generated_call.name if generated_call is not None else None,
        parser_valid=not failures and len(calls) <= 1,
        tool_schema_valid=(
            _tool_schema_valid(generated_call) if generated_call is not None else False
        ),
        bridge_word_count=len(spoken_text.split()),
    )


def generation_metrics(
    predictions: tuple[GenerationPrediction, ...],
) -> GenerationMetrics:
    if not predictions:
        raise ValueError("Generation metrics require at least one prediction.")
    tool_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.case_kind is EvaluationCaseKind.TOOL_CALL
    )
    no_tool_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.case_kind is EvaluationCaseKind.NO_TOOL
    )
    continuation_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.case_kind is EvaluationCaseKind.TOOL_CONTINUATION
    )
    return GenerationMetrics(
        model_variant=predictions[0].model_variant,
        prediction_count=len(predictions),
        parser_valid_rate=_rate(prediction.parser_valid for prediction in predictions),
        tool_decision_accuracy=_rate(
            prediction.generated_tool_name is not None for prediction in tool_predictions
        ),
        tool_name_accuracy=_tool_name_accuracy(tool_predictions),
        tool_schema_valid_rate=_rate(
            prediction.tool_schema_valid for prediction in tool_predictions
        ),
        bridge_present_rate=_rate(
            prediction.bridge_word_count > 0 for prediction in tool_predictions
        ),
        bridge_concise_rate=_rate(
            0 < prediction.bridge_word_count <= 12 for prediction in tool_predictions
        ),
        no_tool_accuracy=_rate(
            prediction.generated_tool_name is None and prediction.parser_valid
            for prediction in no_tool_predictions
        ),
        continuation_nonempty_rate=_rate(
            bool(prediction.spoken_text)
            and prediction.generated_tool_name is None
            and prediction.parser_valid
            for prediction in continuation_predictions
        ),
    )


def _generate_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cases: tuple[HeldoutGenerationCase, ...],
    model_variant: ModelVariant,
    random_seeds: tuple[int, ...],
    batch_size: int,
) -> tuple[GenerationPrediction, ...]:
    model.eval()
    predictions: list[GenerationPrediction] = []
    prompts = tuple(
        cast(
            str,
            tokenizer.apply_chat_template(
                list(case.prefix_messages),
                tools=list(case.tools),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
        )
        for case in cases
    )
    with torch.inference_mode():
        for random_seed in random_seeds:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            for batch_start in range(0, len(cases), batch_size):
                batch_cases = cases[batch_start : batch_start + batch_size]
                batch_prompts = prompts[batch_start : batch_start + batch_size]
                model_inputs = tokenizer(
                    list(batch_prompts),
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
                prompt_width = model_inputs["input_ids"].shape[1]
                for case, output_ids in zip(batch_cases, generated_ids, strict=True):
                    generated_text = tokenizer.decode(
                        output_ids[prompt_width:],
                        skip_special_tokens=True,
                    )
                    predictions.append(
                        parse_generation_prediction(
                            case=case,
                            model_variant=model_variant,
                            random_seed=random_seed,
                            generated_text=generated_text,
                        )
                    )
    return tuple(predictions)


def _tool_schema_valid(call: SerializedToolCall) -> bool:
    try:
        match call.name:
            case ToolName.SEARCH:
                SearchArguments.model_validate_json(call.arguments_json)
            case ToolName.CALCULATE:
                CalculateArguments.model_validate_json(call.arguments_json)
            case ToolName.GET_TIME:
                GetTimeArguments.model_validate_json(call.arguments_json)
            case _:
                return False
    except ValidationError:
        return False
    return True


def _tool_name_accuracy(
    predictions: tuple[GenerationPrediction, ...],
) -> float:
    return _rate(
        prediction.generated_tool_name == prediction.expected_tool_name
        for prediction in predictions
    )


def _rate(values: Iterable[bool]) -> float:
    materialized = tuple(values)
    if not materialized:
        return 0.0
    return sum(bool(value) for value in materialized) / len(materialized)


def _write_predictions(
    path: Path,
    predictions: tuple[GenerationPrediction, ...],
) -> None:
    content = "\n".join(prediction.model_dump_json() for prediction in predictions)
    path.write_text(f"{content}\n", encoding="utf-8")
