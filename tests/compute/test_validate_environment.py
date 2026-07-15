from __future__ import annotations

from deployment.compute import validate_environment


def test_compute_environment_validator_imports_voice_model_constants() -> None:
    assert validate_environment.LANGUAGE_MODEL_NAME == "Qwen/Qwen3-1.7B"
    assert validate_environment.LANGUAGE_MODEL_REVISION
    assert validate_environment.KYUTAI_TTS_MODEL_NAME
    assert validate_environment.KYUTAI_TTS_MODEL_REVISION
