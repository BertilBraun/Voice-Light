from __future__ import annotations

from enum import StrEnum


class VoiceComponent(StrEnum):
    SESSION = "session"
    ASR = "asr"
    LANGUAGE_MODEL = "language_model"
    SPEECH_SYNTHESIS = "speech_synthesis"


class VoiceOperation(StrEnum):
    SESSION_RUN = "session_run"
    TRANSCRIBE = "transcribe"
    GENERATE_TEXT = "generate_text"
    STREAM_SYNTHESIS = "stream_synthesis"


class VoiceComponentError(RuntimeError):
    def __init__(
        self,
        component: VoiceComponent,
        operation: VoiceOperation,
        message: str,
    ) -> None:
        super().__init__(message)
        self.component = component
        self.operation = operation
