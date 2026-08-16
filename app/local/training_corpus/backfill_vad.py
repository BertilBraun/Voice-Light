from __future__ import annotations

import argparse
from uuid import UUID

from app.local.config import DATABASE_URL
from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.db.repository import Repository
from app.local.ingestion.alignment import pad_audio_tracks_to_shared_timeline
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.local.ingestion.local_audio import materialize_sample_track
from app.local.ingestion.vad_repository import VadRepository
from app.shared.audio import load_audio
from app.shared.quality import METRIC_VERSION, TrackVadResult
from app.shared.quality_analysis.preprocessing import prepare_audio_track
from app.shared.quality_analysis.vad import VadConfig, detect_speech_segments_pair
from app.shared.storage.local import LocalStorageBackend

VAD_VERSION = "energy-pair-v1"


def backfill_dataset_vad(
    sample_repository: Repository,
    vad_repository: VadRepository,
    dataset_id: UUID,
) -> int:
    samples = sample_repository.list_annotated_samples(
        dataset_id=dataset_id,
        metric_version=METRIC_VERSION,
        annotation_version=ANNOTATION_VERSION,
        limit=10_000,
        minimum_quality=None,
    )
    backfilled_sample_count = 0
    for index, sample in enumerate(samples, start=1):
        dashboard_sample = sample_repository.get_dashboard_sample(sample.sample_id)
        speaker1_track = track_for_side(dashboard_sample, TrackSide.SPEAKER1)
        speaker2_track = track_for_side(dashboard_sample, TrackSide.SPEAKER2)
        if tracks_have_current_vad(vad_repository, speaker1_track, speaker2_track):
            continue
        speaker1_vad, speaker2_vad = detect_pair_vad(speaker1_track, speaker2_track)
        vad_repository.upsert(
            sample_track_id=speaker1_track.id,
            source_audio_sha256=required_audio_sha256(speaker1_track),
            vad_version=VAD_VERSION,
            result=speaker1_vad,
        )
        vad_repository.upsert(
            sample_track_id=speaker2_track.id,
            source_audio_sha256=required_audio_sha256(speaker2_track),
            vad_version=VAD_VERSION,
            result=speaker2_vad,
        )
        backfilled_sample_count += 1
        print(f"[{index}/{len(samples)}] {sample.external_id}", flush=True)
    return backfilled_sample_count


def tracks_have_current_vad(
    repository: VadRepository,
    speaker1_track: SampleTrackRecord,
    speaker2_track: SampleTrackRecord,
) -> bool:
    return all(
        repository.get_current(
            sample_track_id=track.id,
            source_audio_sha256=required_audio_sha256(track),
            vad_version=VAD_VERSION,
        )
        is not None
        for track in (speaker1_track, speaker2_track)
    )


def detect_pair_vad(
    speaker1_track: SampleTrackRecord,
    speaker2_track: SampleTrackRecord,
) -> tuple[TrackVadResult, TrackVadResult]:
    storage = LocalStorageBackend()
    timeline = pad_audio_tracks_to_shared_timeline(
        speaker1=load_audio(
            storage=storage,
            path=materialize_sample_track(speaker1_track).as_posix(),
            target_sample_rate=16_000,
        ),
        speaker2=load_audio(
            storage=storage,
            path=materialize_sample_track(speaker2_track).as_posix(),
            target_sample_rate=16_000,
        ),
    )
    prepared_speaker1 = prepare_audio_track(timeline.speaker1)
    prepared_speaker2 = prepare_audio_track(timeline.speaker2)
    return detect_speech_segments_pair(
        speaker1_samples=prepared_speaker1.samples,
        speaker2_samples=prepared_speaker2.samples,
        sample_rate=prepared_speaker1.metadata.sample_rate,
        config=VadConfig(),
    )


def track_for_side(
    dashboard_sample: DashboardSample,
    side: TrackSide,
) -> SampleTrackRecord:
    for track in dashboard_sample.tracks:
        if track.side is side:
            return track
    raise ValueError(f"Sample {dashboard_sample.sample.id} has no {side.value} track.")


def required_audio_sha256(track: SampleTrackRecord) -> str:
    if track.audio_sha256 is None:
        raise ValueError(f"Sample track {track.id} has no source audio SHA-256.")
    return track.audio_sha256


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persist the paired energy VAD used by corpus region analysis."
    )
    parser.add_argument("--dataset-id", type=UUID, required=True)
    return parser.parse_args()


def main() -> None:
    options = parse_arguments()
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for VAD backfill.")
    count = backfill_dataset_vad(
        sample_repository=Repository(DATABASE_URL),
        vad_repository=VadRepository(DATABASE_URL),
        dataset_id=options.dataset_id,
    )
    print(f"Backfilled paired VAD for {count} recordings.")


if __name__ == "__main__":
    main()
