from __future__ import annotations

import pytest

from app.compute.voice.tools import ToolName
from deployment.compute.smoke_test_tool_use import (
    SmokeCase,
    _run_smoke_request,
    _smoke_requests,
)


@pytest.mark.parametrize(
    ("case", "generated_text"),
    [
        (SmokeCase.ORDINARY, "I hope your afternoon feels calm and easy."),
        (
            SmokeCase.CALCULATE,
            'I’ll work that out. <tool_call>{"name":"calculate",'
            '"arguments":{"expression":"37 * 14"}}</tool_call>',
        ),
        (
            SmokeCase.SEARCH,
            'I’ll check the latest weather. <tool_call>{"name":"search",'
            '"arguments":{"query":"current weather Berlin"}}</tool_call>',
        ),
        (SmokeCase.POST_TOOL_CONTINUATION, "That comes to 518."),
    ],
)
def test_tool_use_smoke_cases_accept_expected_hermes_behavior(
    case: SmokeCase,
    generated_text: str,
) -> None:
    smoke_request = next(request for request in _smoke_requests() if request.case is case)

    observation = _run_smoke_request(generated_text, smoke_request)

    assert observation.passed
    assert observation.parser_failures == ()


def test_tool_use_smoke_rejects_wrong_tool() -> None:
    smoke_request = next(
        request for request in _smoke_requests() if request.case is SmokeCase.CALCULATE
    )

    observation = _run_smoke_request(
        'I’ll look that up. <tool_call>{"name":"search",'
        '"arguments":{"query":"37 times 14"}}</tool_call>',
        smoke_request,
    )

    assert not observation.passed
    assert observation.generated_tools == (ToolName.SEARCH,)
