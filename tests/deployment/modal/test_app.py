from __future__ import annotations

from pathlib import Path, PurePosixPath

from app.compute.voice.model_constants import (
    SEARCH_SUMMARIZER_MODEL_NAME,
    SEARCH_SUMMARIZER_MODEL_REVISION,
)
from app.shared.model_constants import NEMOTRON_ASR_MODEL_NAME, NEMOTRON_ASR_MODEL_REVISION
from deployment.modal.voice_light import (
    ADAPTER_CHECKPOINT,
    APPLICATION_NAME,
    MERGED_LANGUAGE_MODEL_NAME,
    MERGED_LANGUAGE_MODEL_REVISION,
    MODEL_REPOSITORIES,
    REMOTE_ADAPTER_CHECKPOINT,
    ModalDeploymentConfiguration,
    repository_root,
)


def test_remote_import_uses_packaged_repository_root() -> None:
    assert repository_root(Path("/root/voice_light.py"), local=False) == Path("/opt/voice-light")


def test_adapter_checkpoint_is_packaged_from_expected_backup() -> None:
    assert APPLICATION_NAME == "VoiceLightAgent"
    assert ADAPTER_CHECKPOINT.is_file()
    assert ADAPTER_CHECKPOINT.name == "adapter-best.pt"
    assert ADAPTER_CHECKPOINT.parent.name == "voice-light-human-finetune-v1"


def test_modal_environment_enables_current_voice_stack() -> None:
    configuration = ModalDeploymentConfiguration()
    environment = configuration.environment()

    assert configuration.gpu_options == ("A10:2", "L40S:2")
    assert configuration.compute_regions == ("eu",)
    assert configuration.routing_region == "eu-west"
    assert configuration.compute_secret_name == "voice-light-compute"
    assert configuration.search_secret_name == "voice-light-search"
    assert configuration.scaledown_window_seconds == 120
    assert configuration.startup_timeout_seconds == 1_800
    assert environment["VOICE_LIGHT_EAGER_MODEL_LOADING"] == "true"
    assert environment["VOICE_LIGHT_TRANSCRIPT_FREE_FLOOR_TAKE_DEADLINE_MS"] == "900"
    assert environment["VOICE_LIGHT_MINIMUM_NORMAL_TURN_COMMIT_SILENCE_MS"] == "240"
    assert environment["VOICE_LIGHT_VOICE_STACK_ENABLED"] == "true"
    assert environment["VOICE_LIGHT_TTS_BACKEND"] == "kyutai"
    assert environment["VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS"] == "1"
    assert environment["VOICE_LIGHT_ADAPTER_FIRST_SPECULATION_WINDOW_MS"] == "80"
    assert environment["VOICE_LIGHT_TURN_ADAPTER_CHECKPOINT"] == str(REMOTE_ADAPTER_CHECKPOINT)
    assert environment["VOICE_LIGHT_FLOOR_TAKE_THRESHOLD"] == "0.82"
    assert environment["VOICE_LIGHT_NON_FLOOR_FEEDBACK_THRESHOLD"] == "0.82"
    assert environment["VOICE_LIGHT_OVERLAP_FINALIZATION_GRACE_MS"] == "120"
    assert environment["VOICE_LIGHT_OVERLAP_PREDICTION_SETTLE_MS"] == "80"
    assert environment["VOICE_LIGHT_MAXIMUM_TRANSPORT_AHEAD_MS"] == "1200"
    assert environment["VOICE_LIGHT_MERGED_LANGUAGE_MODEL_NAME"] == MERGED_LANGUAGE_MODEL_NAME
    assert (
        environment["VOICE_LIGHT_MERGED_LANGUAGE_MODEL_REVISION"] == MERGED_LANGUAGE_MODEL_REVISION
    )
    assert environment["VOICE_LIGHT_QWEN_ENFORCE_EAGER"] == "true"
    assert environment["VOICE_LIGHT_QWEN_BACKEND"] == "transformers"
    assert environment["VOICE_LIGHT_QWEN_CUDA_DEVICE"] == "0"
    assert environment["VOICE_LIGHT_QWEN_FIRST_AUDIO_YIELD_ENABLED"] == "false"
    assert environment["VOICE_LIGHT_QWEN_FIRST_AUDIO_YIELD_WORD_COUNT"] == "11"
    assert environment["VOICE_LIGHT_QWEN_FIRST_AUDIO_YIELD_TIMEOUT_MS"] == "400"
    assert environment["VOICE_LIGHT_NEMOTRON_CUDA_DEVICE"] == "1"
    assert environment["VOICE_LIGHT_SEARCH_CUDA_DEVICE"] == "0"
    assert environment["VOICE_LIGHT_TTS_CUDA_DEVICE"] == "1"
    assert environment["VOICE_LIGHT_SHARE_LANGUAGE_MODEL_FOR_SEARCH"] == "false"


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


def test_modal_cache_population_pins_every_hugging_face_repository() -> None:
    repositories_by_id = {repository.repository_id: repository for repository in MODEL_REPOSITORIES}

    assert repositories_by_id[MERGED_LANGUAGE_MODEL_NAME].revision == (
        MERGED_LANGUAGE_MODEL_REVISION
    )
    assert repositories_by_id[NEMOTRON_ASR_MODEL_NAME].revision == NEMOTRON_ASR_MODEL_REVISION
    assert repositories_by_id[SEARCH_SUMMARIZER_MODEL_NAME].revision == (
        SEARCH_SUMMARIZER_MODEL_REVISION
    )
    voice_repository = repositories_by_id["kyutai/tts-voices"]
    assert voice_repository.allowed_files == (
        "expresso/ex03-ex01_happy_001_channel1_334s.wav",
        "expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors",
    )
