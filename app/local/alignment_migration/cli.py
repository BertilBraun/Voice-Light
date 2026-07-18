from __future__ import annotations

import argparse
from enum import StrEnum

from app.local.alignment_migration.repository import AlignmentMigrationRepository
from app.local.alignment_migration.service import AlignmentMigrationService
from app.local.config import DATABASE_URL, SESSIONS_ROOT
from app.local.db.migrate import run_migrations


class MigrationCommand(StrEnum):
    DRY_RUN = "dry-run"
    APPLY = "apply"
    REGENERATE_QUALITY = "regenerate-quality"
    AUDIT = "audit"


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time physical audio-alignment migration.")
    parser.add_argument("command", choices=tuple(MigrationCommand))
    arguments = parser.parse_args()
    command = MigrationCommand(arguments.command)
    repository = AlignmentMigrationRepository(DATABASE_URL)
    service = AlignmentMigrationService(
        repository=repository,
        sessions_root=SESSIONS_ROOT,
    )
    match command:
        case MigrationCommand.DRY_RUN:
            summary = service.dry_run()
        case MigrationCommand.APPLY:
            service.dry_run()
            run_migrations(DATABASE_URL)
            summary = service.apply()
        case MigrationCommand.REGENERATE_QUALITY:
            summary = service.regenerate_quality()
        case MigrationCommand.AUDIT:
            summary = service.audit()
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
