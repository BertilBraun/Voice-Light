from __future__ import annotations

from pathlib import Path

import psycopg

from app.local.config import MIGRATIONS_ROOT


def run_migrations(database_url: str, migrations_root: Path = MIGRATIONS_ROOT) -> None:
    if not database_url:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required to run database migrations.")
    migration_paths = sorted(migrations_root.glob("*.sql"))
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                  version text PRIMARY KEY,
                  applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            for migration_path in migration_paths:
                version = migration_path.name
                cursor.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
                if cursor.fetchone() is not None:
                    continue
                cursor.execute(migration_path.read_text(encoding="utf-8"))
                cursor.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))


def main() -> None:
    from app.local.config import DATABASE_URL

    run_migrations(DATABASE_URL)


if __name__ == "__main__":
    main()
