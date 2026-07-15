from __future__ import annotations

from app.compute.voice.models import _run_generation


class RecordingGenerationStreamer:
    def __init__(self) -> None:
        self.finalized: list[tuple[str, bool]] = []

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        self.finalized.append((text, stream_end))


def test_generation_failure_ends_stream_and_preserves_error() -> None:
    expected_error = RuntimeError("CUDA failed")
    streamer = RecordingGenerationStreamer()
    generation_errors: list[Exception] = []

    def fail_generation() -> None:
        raise expected_error

    _run_generation(fail_generation, streamer, generation_errors)

    assert generation_errors == [expected_error]
    assert streamer.finalized == [("", True)]
