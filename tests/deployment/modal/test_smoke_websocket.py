from __future__ import annotations

import asyncio

import pytest

from deployment.modal.smoke_websocket import run_smoke


def test_smoke_rejects_non_positive_open_timeout() -> None:
    with pytest.raises(ValueError, match="WebSocket open timeout must be positive"):
        asyncio.run(run_smoke("wss://example.invalid/v1/voice", 0.0))
