from __future__ import annotations

import threading

import pytest

from app.compute.voice.subprocess_start import read_worker_start_event


def test_worker_start_event_is_returned() -> None:
    assert read_worker_start_event(lambda: "ready", lambda: None, 1.0, "Test") == "ready"


def test_worker_start_failure_preserves_cause() -> None:
    expected_error = RuntimeError("synthetic startup failure")

    def fail() -> str:
        raise expected_error

    with pytest.raises(RuntimeError, match="Test worker startup failed") as raised:
        read_worker_start_event(fail, lambda: None, 1.0, "Test")

    assert raised.value.__cause__ is expected_error


def test_worker_start_timeout_terminates_process() -> None:
    terminated = threading.Event()

    def wait_for_termination() -> str:
        terminated.wait()
        return "stopped"

    with pytest.raises(RuntimeError, match="Test worker startup timed out"):
        read_worker_start_event(wait_for_termination, terminated.set, 0.01, "Test")

    assert terminated.is_set()
