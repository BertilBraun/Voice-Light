from __future__ import annotations

import queue
import threading
from typing import TypeVar

QueueItemType = TypeVar("QueueItemType")


def get_until_cancelled(
    items: queue.Queue[QueueItemType],
    cancellation_event: threading.Event,
    poll_interval_seconds: float,
) -> QueueItemType | None:
    if poll_interval_seconds <= 0:
        raise ValueError("Queue poll interval must be positive.")
    while not cancellation_event.is_set():
        try:
            return items.get(timeout=poll_interval_seconds)
        except queue.Empty:
            continue
    return None
