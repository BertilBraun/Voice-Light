from __future__ import annotations

import pytest

from app.compute.voice.admission import SingleVoiceSessionAdmission, VoiceSessionLease


def test_only_one_voice_session_is_admitted() -> None:
    admission = SingleVoiceSessionAdmission()

    first_lease = admission.try_acquire("first")

    assert first_lease == VoiceSessionLease(request_id="first")
    assert admission.active_request_id == "first"
    assert admission.try_acquire("second") is None


def test_voice_session_can_be_admitted_after_release() -> None:
    admission = SingleVoiceSessionAdmission()
    first_lease = admission.try_acquire("first")
    assert first_lease is not None

    admission.release(first_lease)

    assert admission.active_request_id is None
    assert admission.try_acquire("second") == VoiceSessionLease(request_id="second")


def test_wrong_voice_session_lease_cannot_be_released() -> None:
    admission = SingleVoiceSessionAdmission()
    active_lease = admission.try_acquire("active")
    assert active_lease is not None

    with pytest.raises(AssertionError, match="not active"):
        admission.release(VoiceSessionLease(request_id="other"))

    assert admission.active_request_id == "active"
