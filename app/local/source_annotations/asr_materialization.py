from __future__ import annotations

from pathlib import Path

from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.source_annotations.models import SourceAnnotationTrack, SourceSampleAnnotations
from app.shared.asr import AsrModelId, AsrTranscriptResult, TimestampedWord

SOURCE_TRANSCRIPT_ANNOTATOR_ID = "a"
SOURCE_TRANSCRIPT_MODELS = (AsrModelId.PARAKEET_TDT, AsrModelId.CANARY)


def materialize_source_transcripts(
    repository: FullRecordingAsrRepository,
    dashboard_sample: DashboardSample,
    annotations: SourceSampleAnnotations,
) -> None:
    for track, annotation_track in (
        (sample_track_for_side(dashboard_sample, TrackSide.SPEAKER1), annotations.speaker_1),
        (sample_track_for_side(dashboard_sample, TrackSide.SPEAKER2), annotations.speaker_2),
    ):
        source_track = source_transcript_track(annotation_track)
        transcript_words = timestamped_source_events(source_track)
        transcript_text = " ".join(word.text for word in transcript_words)
        for model_id in SOURCE_TRANSCRIPT_MODELS:
            repository.upsert_transcript(
                sample_track_id=track.id,
                source_audio_sha256=audio_sha256(track),
                prepared_audio_sha256=audio_sha256(track),
                audio_filename=Path(track.storage_uri).name,
                source_duration_seconds=duration_seconds(track),
                prepared_duration_seconds=duration_seconds(track),
                transcript=AsrTranscriptResult(
                    model_id=model_id,
                    text=transcript_text,
                    words=transcript_words,
                    language_estimate=None,
                ),
            )


def source_transcript_track(
    annotation_tracks: tuple[SourceAnnotationTrack, ...],
) -> SourceAnnotationTrack:
    for annotation_track in annotation_tracks:
        if annotation_track.annotator_id == SOURCE_TRANSCRIPT_ANNOTATOR_ID:
            return annotation_track
    raise ValueError(
        f"Source annotations do not include annotator {SOURCE_TRANSCRIPT_ANNOTATOR_ID!r}."
    )


def timestamped_source_events(
    annotation_track: SourceAnnotationTrack,
) -> tuple[TimestampedWord, ...]:
    return tuple(
        TimestampedWord(
            text=event.text.strip(),
            start_seconds=event.start_seconds,
            end_seconds=event.end_seconds,
        )
        for event in annotation_track.events
        if event.text.strip()
    )


def sample_track_for_side(
    dashboard_sample: DashboardSample,
    side: TrackSide,
) -> SampleTrackRecord:
    for track in dashboard_sample.tracks:
        if track.side is side:
            return track
    raise ValueError(f"Sample does not have a {side.value} track.")


def audio_sha256(track: SampleTrackRecord) -> str:
    if track.audio_sha256 is None:
        raise ValueError(f"Sample track {track.id} does not have an audio SHA-256.")
    return track.audio_sha256


def duration_seconds(track: SampleTrackRecord) -> float:
    if track.duration_seconds is None:
        raise ValueError(f"Sample track {track.id} does not have a duration.")
    return track.duration_seconds
