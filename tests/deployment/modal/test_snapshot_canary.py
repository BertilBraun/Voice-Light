from __future__ import annotations

from deployment.modal.voice_light import APPLICATION_NAME, ModalDeploymentConfiguration
from deployment.modal.voice_light_snapshot_canary import (
    CANARY_APPLICATION_NAME,
    CANARY_ENDPOINT_LABEL,
    CANARY_GPU,
    app,
)


def test_snapshot_canary_is_isolated_on_a_fixed_supported_gpu() -> None:
    production_configuration = ModalDeploymentConfiguration()

    assert app.name == CANARY_APPLICATION_NAME
    assert CANARY_APPLICATION_NAME != APPLICATION_NAME
    assert CANARY_ENDPOINT_LABEL == "voicelightagent-voice-light-snapshot-canary"
    assert CANARY_GPU == "A10"
    assert CANARY_GPU in production_configuration.gpu_options
