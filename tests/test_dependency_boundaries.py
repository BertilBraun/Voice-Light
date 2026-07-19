from __future__ import annotations

from pathlib import Path

import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
COMPUTE_ONLY_PACKAGES = (
    "faster-whisper",
    "librosa",
    "nemo-toolkit",
    "moshi",
)


def test_compute_dependencies_are_excluded_from_local_install() -> None:
    project_configuration = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = project_configuration["project"]
    local_dependencies = tuple(project["dependencies"])
    compute_dependencies = tuple(project["optional-dependencies"]["compute"])

    for package_name in COMPUTE_ONLY_PACKAGES:
        assert not contains_package(local_dependencies, package_name)
        assert contains_package(compute_dependencies, package_name)


def test_compute_deployment_installs_compute_extra() -> None:
    for relative_path in (
        Path("deployment/compute/bootstrap.sh"),
        Path("deployment/compute/start.sh"),
    ):
        script = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        assert "uv sync --frozen --python 3.12 --extra compute" in script


def test_voxtream_uses_a_pinned_isolated_environment() -> None:
    bootstrap_script = (REPOSITORY_ROOT / "deployment/compute/bootstrap.sh").read_text(
        encoding="utf-8"
    )
    installation_script = (REPOSITORY_ROOT / "deployment/compute/install_voxtream.sh").read_text(
        encoding="utf-8"
    )

    assert 'VOICE_LIGHT_TTS_BACKEND:-kyutai}" == "voxtream"' in bootstrap_script
    assert "8ec2d62159dae4716ae7058827244a962d40603c" in installation_script
    assert 'voxtream_root="${repository_root}/.cache/compute/voxtream"' in installation_script
    assert 'python_path="${voxtream_root}/.venv/bin/python"' in installation_script
    assert "voxtream-prompt-memory-cache.patch" in installation_script
    assert "voxtream-optional-speaking-rate.patch" in installation_script
    assert 'git -C "$source_root" apply' in installation_script
    assert "uv pip install" in installation_script


def test_vast_provisioning_installs_supervisor_service() -> None:
    provisioning_script = (REPOSITORY_ROOT / "deployment/compute/provision-vast.sh").read_text(
        encoding="utf-8"
    )
    service_script = (REPOSITORY_ROOT / "deployment/compute/install-service.sh").read_text(
        encoding="utf-8"
    )

    stop_index = provisioning_script.index("supervisorctl stop voice-light-compute")
    bootstrap_index = provisioning_script.index("bash deployment/compute/bootstrap.sh")
    assert stop_index < bootstrap_index
    assert "bash deployment/compute/bootstrap.sh" in provisioning_script
    assert "bash deployment/compute/install-service.sh" in provisioning_script
    assert "supervisorctl update" in service_script


def test_vast_deployment_waits_for_authenticated_model_readiness() -> None:
    deployment_script = (REPOSITORY_ROOT / "deployment/compute/deploy-vast.ps1").read_text(
        encoding="utf-8"
    )

    assert "/health/ready" in deployment_script
    assert "Authorization" in deployment_script
    assert "/health/live" not in deployment_script


def test_compute_lifecycle_commands_support_supervisor() -> None:
    for script_name in ("start.sh", "status.sh", "stop.sh"):
        script = (REPOSITORY_ROOT / "deployment/compute" / script_name).read_text(encoding="utf-8")
        assert "supervisorctl" in script


def contains_package(dependencies: tuple[str, ...], package_name: str) -> bool:
    return any(dependency.startswith(package_name) for dependency in dependencies)
