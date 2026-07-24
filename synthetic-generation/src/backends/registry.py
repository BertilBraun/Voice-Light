from __future__ import annotations

from collections.abc import Callable

from src.backends.base import TTSBackend
from src.backends.elevenlabs import ElevenLabsBackend
from src.backends.fake import FakeBackend
from src.backends.qwen_voice_design import QwenVoiceDesignBackend

BackendFactory = Callable[[str, dict[str, object]], TTSBackend]

BACKENDS: dict[str, BackendFactory] = {
    "qwen-voice-design": QwenVoiceDesignBackend,
    "elevenlabs": ElevenLabsBackend,
    "fake": FakeBackend,
}


def create_backend(backend: str, model: str, options: dict[str, object]) -> TTSBackend:
    try:
        backend_factory = BACKENDS[backend]
    except KeyError as error:
        available_backends = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend '{backend}'. Available: {available_backends}") from error
    return backend_factory(model, options)
