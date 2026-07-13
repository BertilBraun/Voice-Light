from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = REPOSITORY_ROOT / "app"


def imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    modules: list[str] = []
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                modules.extend(alias.name for alias in names)
            case ast.ImportFrom(module=module) if module is not None:
                modules.append(module)
    return tuple(modules)


def python_files(package_name: str) -> tuple[Path, ...]:
    return tuple(sorted((APP_ROOT / package_name).rglob("*.py")))


def test_app_root_contains_only_deployable_packages() -> None:
    package_directories = {
        path.name for path in APP_ROOT.iterdir() if path.is_dir() and path.name != "__pycache__"
    }

    assert package_directories == {"compute", "local", "shared", "training"}


@pytest.mark.parametrize(
    ("package_name", "forbidden_prefixes"),
    [
        ("compute", ("app.local",)),
        ("local", ("app.compute",)),
        ("shared", ("app.compute", "app.local", "app.training")),
    ],
)
def test_runtime_packages_respect_deployment_boundaries(
    package_name: str,
    forbidden_prefixes: tuple[str, ...],
) -> None:
    violations = [
        f"{path.relative_to(REPOSITORY_ROOT)} imports {module}"
        for path in python_files(package_name)
        for module in imported_modules(path)
        if module.startswith(forbidden_prefixes)
    ]

    assert violations == []
