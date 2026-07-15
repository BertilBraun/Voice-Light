from __future__ import annotations

import queue
import threading

from app.compute.voice.thread_queue import get_until_cancelled


def test_empty_queue_poll_stops_when_cancellation_arrives() -> None:
    items: queue.Queue[int] = queue.Queue()
    cancellation_event = threading.Event()
    results: list[int | None] = []

    def wait_for_item() -> None:
        results.append(get_until_cancelled(items, cancellation_event, 0.01))

    worker = threading.Thread(target=wait_for_item)
    worker.start()
    cancellation_event.set()
    worker.join(timeout=0.5)

    assert not worker.is_alive()
    assert results == [None]


def test_queue_item_wins_before_cancellation() -> None:
    items: queue.Queue[int] = queue.Queue()
    cancellation_event = threading.Event()
    items.put(42)

    assert get_until_cancelled(items, cancellation_event, 0.01) == 42
