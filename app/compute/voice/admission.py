from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceSessionLease:
    request_id: str


class SingleVoiceSessionAdmission:
    def __init__(self) -> None:
        self._active_lease: VoiceSessionLease | None = None

    @property
    def active_request_id(self) -> str | None:
        if self._active_lease is None:
            return None
        return self._active_lease.request_id

    def try_acquire(self, request_id: str) -> VoiceSessionLease | None:
        if not request_id:
            raise ValueError("Voice session request ID must not be empty.")
        if self._active_lease is not None:
            return None
        lease = VoiceSessionLease(request_id=request_id)
        self._active_lease = lease
        return lease

    def release(self, lease: VoiceSessionLease) -> None:
        if self._active_lease is not lease:
            raise AssertionError("Cannot release a voice session lease that is not active.")
        self._active_lease = None
