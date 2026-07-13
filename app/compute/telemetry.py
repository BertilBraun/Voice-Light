from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

import torch

from app.shared.compute_api import GpuMemory

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str | int | float | None] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_context.get(),
        }
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_directory: Path) -> None:
    log_directory.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()
    file_handler = logging.FileHandler(log_directory / "compute.jsonl", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)


class RequestIdScope:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.token: Token[str | None] | None = None

    def __enter__(self) -> RequestIdScope:
        self.token = request_id_context.set(self.request_id)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        assert self.token is not None
        request_id_context.reset(self.token)


def gpu_memory() -> GpuMemory:
    if not torch.cuda.is_available():
        return GpuMemory(current_allocated_mb=None, peak_allocated_mb=None)
    return GpuMemory(
        current_allocated_mb=torch.cuda.memory_allocated() / 1_048_576,
        peak_allocated_mb=torch.cuda.max_memory_allocated() / 1_048_576,
    )
