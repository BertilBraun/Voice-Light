from __future__ import annotations

from pathlib import PurePosixPath

from deployment.modal.app import (
    ADAPTER_CHECKPOINT,
    APPLICATION_NAME,
    REMOTE_ADAPTER_CHECKPOINT,
    ModalDeploymentConfiguration,
)


def test_adapter_checkpoint_is_packaged_from_expected_backup() -> None:
    assert APPLICATION_NAME == "VoiceLightAgent"
    assert ADAPTER_CHECKPOINT.is_file()
    assert ADAPTER_CHECKPOINT.name == "adapter-best.pt"
    assert ADAPTER_CHECKPOINT.parent.name == "voice-light-human-finetune-v1"


def test_modal_environment_enables_current_voice_stack() -> None:
    environment = ModalDeploymentConfiguration().environment()

    assert environment["VOICE_LIGHT_VOICE_STACK_ENABLED"] == "true"
    assert environment["VOICE_LIGHT_TTS_BACKEND"] == "kyutai"
    assert environment["VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS"] == "1"
    assert environment["VOICE_LIGHT_TURN_ADAPTER_CHECKPOINT"] == str(REMOTE_ADAPTER_CHECKPOINT)
    assert environment["VOICE_LIGHT_FLOOR_TAKE_THRESHOLD"] == "0.82"
    assert environment["VOICE_LIGHT_NON_FLOOR_FEEDBACK_THRESHOLD"] == "0.82"


def test_modal_cache_paths_are_absolute() -> None:
    environment = ModalDeploymentConfiguration().environment()

    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "TORCH_HOME",
        "VOICE_LIGHT_COMPUTE_LOG_DIR",
        "VOICE_LIGHT_DATASET_AUDIO_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        assert PurePosixPath(environment[name]).is_absolute()
