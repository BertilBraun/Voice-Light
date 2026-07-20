from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from app.local.config import DATABASE_URL
from app.local.ingestion.manifest import read_meetings_manifest


@dataclass(frozen=True)
class PruneCandidate:
    sample_id: UUID
    external_id: str


def prune_dataset_to_manifest(
    database_url: str,
    dataset_id: UUID,
    retained_external_ids: frozenset[str],
    apply: bool,
) -> tuple[PruneCandidate, ...]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows = connection.execute(
            """
            SELECT id, external_id
            FROM samples
            WHERE dataset_id = %s
            ORDER BY external_id
            """,
            (dataset_id,),
        ).fetchall()
        candidates = tuple(
            PruneCandidate(
                sample_id=sample_id_from_row(row["id"]),
                external_id=external_id_from_row(row["external_id"]),
            )
            for row in rows
            if external_id_from_row(row["external_id"]) not in retained_external_ids
        )
        if apply and candidates:
            connection.execute(
                "DELETE FROM samples WHERE id = ANY(%s)",
                ([candidate.sample_id for candidate in candidates],),
            )
    return candidates


def sample_id_from_row(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def external_id_from_row(value: object) -> str:
    assert isinstance(value, str)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=UUID, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    options = parser.parse_args()
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required.")
    manifest = read_meetings_manifest(options.manifest_path.resolve())
    candidates = prune_dataset_to_manifest(
        database_url=DATABASE_URL,
        dataset_id=options.dataset_id,
        retained_external_ids=frozenset(sample.sample_path for sample in manifest.samples),
        apply=options.apply,
    )
    action = "Deleted" if options.apply else "Would delete"
    print(f"{action} {len(candidates)} samples absent from the manifest.")
    for candidate in candidates:
        print(candidate.external_id)


if __name__ == "__main__":
    main()
