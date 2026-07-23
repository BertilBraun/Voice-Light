from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.source_annotations.models import (
    RegisteredSourceAnnotationSample,
    SourceAnnotationEvent,
    SourceAnnotationImport,
    SourceAnnotationTrack,
)


class SourceAnnotationRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def import_after_sample_registration(
        self, annotations: Sequence[SourceAnnotationImport]
    ) -> None:
        if not annotations:
            return
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            for annotation in annotations:
                connection.execute(
                    """
                    INSERT INTO source_track_annotations (
                      sample_track_id, source_name, annotation_version, annotator_id, events,
                      updated_at
                    ) VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT (sample_track_id, source_name, annotation_version, annotator_id)
                    DO UPDATE SET events = EXCLUDED.events, updated_at = now()
                    """,
                    (
                        annotation.sample_track_id,
                        annotation.source_name,
                        annotation.annotation_version,
                        annotation.annotator_id,
                        Jsonb(
                            [
                                event.model_dump(mode="json")
                                for event in annotation.annotation.events
                            ]
                        ),
                    ),
                )

    def list_for_sample(self, sample_id: UUID) -> tuple[SourceAnnotationImport, ...]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT annotations.sample_track_id, annotations.source_name,
                       annotations.annotation_version, annotations.annotator_id,
                       annotations.events
                FROM source_track_annotations AS annotations
                JOIN sample_tracks ON sample_tracks.id = annotations.sample_track_id
                WHERE sample_tracks.sample_id = %s
                ORDER BY sample_tracks.speaker_index, annotations.annotator_id
                """,
                (sample_id,),
            ).fetchall()
        return tuple(
            SourceAnnotationImport(
                sample_track_id=UUID(str(row["sample_track_id"])),
                source_name=str(row["source_name"]),
                annotation_version=str(row["annotation_version"]),
                annotator_id=str(row["annotator_id"]),
                annotation=annotation_track_from_row(
                    annotator_id=str(row["annotator_id"]), events=row["events"]
                ),
            )
            for row in rows
        )

    def list_registered_samples(
        self, dataset_name: str
    ) -> tuple[RegisteredSourceAnnotationSample, ...]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT samples.id AS sample_id, samples.external_id,
                       speaker_1.id AS speaker_1_track_id,
                       speaker_2.id AS speaker_2_track_id
                FROM datasets
                JOIN samples ON samples.dataset_id = datasets.id
                JOIN sample_tracks AS speaker_1
                  ON speaker_1.sample_id = samples.id AND speaker_1.speaker_index = 1
                JOIN sample_tracks AS speaker_2
                  ON speaker_2.sample_id = samples.id AND speaker_2.speaker_index = 2
                WHERE datasets.name = %s
                ORDER BY samples.external_id
                """,
                (dataset_name,),
            ).fetchall()
        return tuple(
            RegisteredSourceAnnotationSample(
                sample_id=UUID(str(row["sample_id"])),
                external_id=str(row["external_id"]),
                speaker_1_track_id=UUID(str(row["speaker_1_track_id"])),
                speaker_2_track_id=UUID(str(row["speaker_2_track_id"])),
            )
            for row in rows
        )


def annotation_track_from_row(annotator_id: str, events: object) -> SourceAnnotationTrack:
    if not isinstance(events, list):
        raise ValueError("Stored source annotation events must be a JSON array.")
    return SourceAnnotationTrack(
        annotator_id=annotator_id,
        events=tuple(SourceAnnotationEvent.model_validate(event) for event in events),
    )
