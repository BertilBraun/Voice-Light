from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.local.alignment_migration import cli
from app.local.alignment_migration.models import MigrationQualitySummary


class FakeMigrationService:
    calls: list[str] = []
    events: list[str] = []

    def __init__(self, repository: str, sessions_root: Path) -> None:
        assert repository == "repository"
        assert sessions_root == cli.SESSIONS_ROOT

    def dry_run(self) -> MigrationQualitySummary:
        self.calls.append("dry-run")
        self.events.append("dry-run")
        return summary()

    def apply(self) -> MigrationQualitySummary:
        self.calls.append("apply")
        self.events.append("apply")
        return summary()

    def regenerate_quality(self) -> MigrationQualitySummary:
        self.calls.append("regenerate-quality")
        self.events.append("regenerate-quality")
        return summary()

    def audit(self) -> MigrationQualitySummary:
        self.calls.append("audit")
        self.events.append("audit")
        return summary()


@pytest.mark.parametrize(
    "command",
    ("dry-run", "apply", "regenerate-quality", "audit"),
)
def test_cli_dispatches_every_migration_command(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    FakeMigrationService.calls = []
    FakeMigrationService.events = events
    monkeypatch.setattr(
        cli,
        "AlignmentMigrationRepository",
        lambda database_url: "repository",
    )
    monkeypatch.setattr(cli, "AlignmentMigrationService", FakeMigrationService)
    monkeypatch.setattr(
        cli,
        "run_migrations",
        lambda database_url: events.append("migrations"),
    )
    monkeypatch.setattr(sys, "argv", ["alignment-migration", command])

    cli.main()

    assert FakeMigrationService.calls == (["dry-run", "apply"] if command == "apply" else [command])
    assert events == (["dry-run", "migrations", "apply"] if command == "apply" else [command])
    assert '"regenerated_count": 0' in capsys.readouterr().out


def summary() -> MigrationQualitySummary:
    return MigrationQualitySummary(
        retained_sample_count=165,
        already_current_count=165,
        regenerated_count=0,
    )
