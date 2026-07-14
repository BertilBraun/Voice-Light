from __future__ import annotations

from pathlib import Path

import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
COMPUTE_ONLY_PACKAGES = (
    "faster-whisper",
    "librosa",
    "nemo-toolkit",
    "pocket-tts",
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


def contains_package(dependencies: tuple[str, ...], package_name: str) -> bool:
    return any(dependency.startswith(package_name) for dependency in dependencies)
