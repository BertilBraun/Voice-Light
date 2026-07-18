from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from app.training.tool_use.protocol import (
    GeneratedUserTurn,
    GeneratedUserTurnEnvelope,
    TeacherChatMessage,
)
from app.training.tool_use.schema import SpeechStyle
from app.training.tool_use.vllm_client import (
    StructuredGenerationResult,
    VllmClientConfig,
    VllmStructuredClient,
)


def test_vllm_client_sends_json_schema_and_disables_thinking(tmp_path: Path) -> None:
    captured_request: httpx.Request | None = None
    request_log_path = tmp_path / "requests.jsonl"

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        content = GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="what time is it",
                speech_style=SpeechStyle.CASUAL,
            )
        ).model_dump_json()
        return httpx.Response(
            status_code=200,
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 1_784_363_084,
                "model": "Qwen/Qwen3.6-27B-FP8",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "refusal": None,
                            "annotations": None,
                            "audio": None,
                            "function_call": None,
                            "tool_calls": [],
                            "reasoning": None,
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                        "stop_reason": None,
                        "token_ids": None,
                    }
                ],
                "service_tier": None,
                "system_fingerprint": None,
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                    "prompt_tokens_details": None,
                },
                "prompt_logprobs": None,
                "prompt_token_ids": None,
                "kv_transfer_params": None,
            },
        )

    async def exercise() -> StructuredGenerationResult[GeneratedUserTurnEnvelope]:
        client = VllmStructuredClient(
            config=VllmClientConfig(
                base_url="http://vllm.test/v1",
                api_key="test-key",
                model_identifier="Qwen/Qwen3.6-27B-FP8",
            ),
            request_log_path=request_log_path,
            transport=httpx.MockTransport(handler),
        )
        async with client:
            return await client.generate(
                messages=(TeacherChatMessage(role="user", content="Create a user turn."),),
                response_type=GeneratedUserTurnEnvelope,
                random_seed=17,
            )

    result = asyncio.run(exercise())
    assert result.value.turn.text == "what time is it"
    assert result.prompt_tokens == 12
    assert captured_request is not None
    payload = json.loads(captured_request.content)
    assert payload["response_format"]["type"] == "json_schema"
    assert "schema" in payload["response_format"]["json_schema"]
    assert "schema_value" not in payload["response_format"]["json_schema"]
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["seed"] == 17
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repetition_penalty"] == 1.0
    log_entries = request_log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_entries) == 1
    log_entry = json.loads(log_entries[0])
    assert log_entry["request_body"] == captured_request.content.decode()
    assert log_entry["status_code"] == 200
    assert log_entry["response_body"] is not None
    assert "Authorization" not in log_entries[0]
