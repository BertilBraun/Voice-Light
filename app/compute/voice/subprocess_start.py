from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

WorkerEvent = TypeVar("WorkerEvent")


@dataclass(frozen=True)
class _StartupEvent(Generic[WorkerEvent]):
    event: WorkerEvent


@dataclass(frozen=True)
class _StartupFailure:
    error: Exception


def read_worker_start_event(
    read_event: Callable[[], WorkerEvent],
    terminate: Callable[[], None],
    timeout_seconds: float,
    component_name: str,
) -> WorkerEvent:
    if timeout_seconds <= 0:
        raise ValueError("Worker startup timeout must be positive.")
    result_queue: queue.Queue[_StartupEvent[WorkerEvent] | _StartupFailure] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result_queue.put(_StartupEvent(read_event()))
        except Exception as error:
            result_queue.put(_StartupFailure(error))

    reader_thread = threading.Thread(
        target=read,
        daemon=True,
        name=f"{component_name}-startup-reader",
    )
    reader_thread.start()
    try:
        result = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as error:
        terminate()
        reader_thread.join(timeout=1.0)
        raise RuntimeError(f"{component_name} worker startup timed out.") from error
    match result:
        case _StartupEvent(event=event):
            return event
        case _StartupFailure(error=error):
            terminate()
            raise RuntimeError(f"{component_name} worker startup failed.") from error
