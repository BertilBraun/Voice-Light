from __future__ import annotations

import argparse
from enum import StrEnum

from app.local.config import DATABASE_URL, SESSIONS_ROOT
from app.local.db.migrate import run_migrations
from app.local.timeline_repair.materialize import TimelineMaterializationService


class MaterializationCommand(StrEnum):
    DRY_RUN = "dry-run"
    APPLY = "apply"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize approved timeline repairs.")
    parser.add_argument("command", choices=tuple(MaterializationCommand))
    arguments = parser.parse_args()
    command = MaterializationCommand(arguments.command)
    run_migrations(DATABASE_URL)
    service = TimelineMaterializationService(
        database_url=DATABASE_URL,
        sessions_root=SESSIONS_ROOT,
    )
    if command is MaterializationCommand.DRY_RUN:
        summary = service.dry_run()
    else:
        service.dry_run()
        summary = service.apply()
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
