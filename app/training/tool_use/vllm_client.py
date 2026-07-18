from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Generic, Literal, TypeVar

import httpx
from pydantic import Field, JsonValue

from app.training.tool_use.protocol import TeacherChatMessage
from app.training.tool_use.schema import ToolUseBaseModel

StructuredOutput = TypeVar("StructuredOutput", bound=ToolUseBaseModel)


class VllmClientConfig(ToolUseBaseModel):
    base_url: str
    api_key: str
    model_identifier: str
    request_timeout_seconds: float = Field(default=180.0, gt=0.0)
    maximum_http_attempts: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=1.5, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    maximum_tokens: int = Field(default=1200, ge=1)


class JsonSchemaSpecification(ToolUseBaseModel):
    name: str
    strict: Literal[True] = True
    schema_value: dict[str, JsonValue] = Field(alias="schema")


class JsonSchemaResponseFormat(ToolUseBaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: JsonSchemaSpecification


class ChatTemplateArguments(ToolUseBaseModel):
    enable_thinking: bool = False


class VllmChatCompletionRequest(ToolUseBaseModel):
    model: str
    messages: tuple[TeacherChatMessage, ...]
    temperature: float
    top_p: float
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_tokens: int
    seed: int
    response_format: JsonSchemaResponseFormat
    chat_template_kwargs: ChatTemplateArguments
    top_k: int


class VllmResponseMessage(ToolUseBaseModel):
    role: Literal["assistant"]
    content: str | None
    refusal: str | None
    annotations: tuple[JsonValue, ...] | None
    audio: JsonValue | None
    function_call: JsonValue | None
    tool_calls: tuple[JsonValue, ...]
    reasoning: str | None


class VllmResponseChoice(ToolUseBaseModel):
    index: int
    message: VllmResponseMessage
    finish_reason: str | None
    logprobs: JsonValue | None
    stop_reason: str | int | None
    token_ids: tuple[int, ...] | None


class VllmUsage(ToolUseBaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    prompt_tokens_details: JsonValue | None


class VllmChatCompletionResponse(ToolUseBaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int = Field(ge=0)
    model: str
    choices: tuple[VllmResponseChoice, ...]
    service_tier: str | None
    system_fingerprint: str | None
    usage: VllmUsage | None = None
    prompt_logprobs: JsonValue | None
    prompt_token_ids: tuple[int, ...] | None
    kv_transfer_params: JsonValue | None


class VllmRequestLogEntry(ToolUseBaseModel):
    recorded_at: str
    attempt: int
    request_body: str
    status_code: int | None
    response_body: str | None
    error_message: str | None


class StructuredGenerationResult(ToolUseBaseModel, Generic[StructuredOutput]):
    value: StructuredOutput
    prompt_tokens: int
    completion_tokens: int


class VllmStructuredClient:
    def __init__(
        self,
        config: VllmClientConfig,
        request_log_path: Path | None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.request_log_path = request_log_path
        self.request_log_lock = asyncio.Lock()
        self.client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.request_timeout_seconds,
            transport=transport,
        )

    async def __aenter__(self) -> VllmStructuredClient:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self.client.aclose()

    async def generate(
        self,
        messages: Sequence[TeacherChatMessage],
        response_type: type[StructuredOutput],
        random_seed: int,
    ) -> StructuredGenerationResult[StructuredOutput]:
        schema_value: dict[str, JsonValue] = response_type.model_json_schema()
        request = VllmChatCompletionRequest(
            model=self.config.model_identifier,
            messages=tuple(messages),
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            min_p=self.config.min_p,
            presence_penalty=self.config.presence_penalty,
            repetition_penalty=self.config.repetition_penalty,
            max_tokens=self.config.maximum_tokens,
            seed=random_seed,
            response_format=JsonSchemaResponseFormat(
                json_schema=JsonSchemaSpecification(
                    name=response_type.__name__,
                    schema_value=schema_value,
                )
            ),
            chat_template_kwargs=ChatTemplateArguments(),
            top_k=self.config.top_k,
        )
        response = await self._post_with_retry(request)
        if len(response.choices) != 1:
            raise ValueError(f"Expected one vLLM choice, received {len(response.choices)}.")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("vLLM returned no structured response content.")
        value = response_type.model_validate_json(content)
        usage = response.usage
        return StructuredGenerationResult[StructuredOutput](
            value=value,
            prompt_tokens=usage.prompt_tokens if usage is not None else 0,
            completion_tokens=usage.completion_tokens if usage is not None else 0,
        )

    async def _post_with_retry(
        self,
        request: VllmChatCompletionRequest,
    ) -> VllmChatCompletionResponse:
        request_body = request.model_dump_json(by_alias=True)
        last_error: Exception | None = None
        for attempt in range(1, self.config.maximum_http_attempts + 1):
            response: httpx.Response | None = None
            try:
                response = await self.client.post(
                    "chat/completions",
                    content=request_body,
                    headers={"Content-Type": "application/json"},
                )
                await self._append_request_log(
                    VllmRequestLogEntry(
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                        attempt=attempt,
                        request_body=request_body,
                        status_code=response.status_code,
                        response_body=response.text,
                        error_message=(
                            None if response.is_success else f"HTTP status {response.status_code}"
                        ),
                    )
                )
                response.raise_for_status()
                return VllmChatCompletionResponse.model_validate_json(response.text)
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if response is None:
                    await self._append_request_log(
                        VllmRequestLogEntry(
                            recorded_at=datetime.now(timezone.utc).isoformat(),
                            attempt=attempt,
                            request_body=request_body,
                            status_code=None,
                            response_body=None,
                            error_message=str(error),
                        )
                    )
                if attempt < self.config.maximum_http_attempts:
                    await asyncio.sleep(0.25 * attempt)
        assert last_error is not None
        raise last_error

    async def _append_request_log(self, entry: VllmRequestLogEntry) -> None:
        if self.request_log_path is None:
            return

        async with self.request_log_lock:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.request_log_path.open("a", encoding="utf-8") as request_log:
                request_log.write(entry.model_dump_json())
                request_log.write("\n")
