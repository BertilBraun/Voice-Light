from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from uuid import UUID

from app.local.analyses.end_of_turn.detectors.naive_vad import run_naive_vad_floor_full_recording
from app.local.analyses.end_of_turn.service import SpeechSegment as BaselineSpeechSegment
from app.local.config import DATABASE_URL
from app.local.corpus_audit.repository import CorpusAuditRepository
from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.db.repository import Repository
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.local.ingestion.vad_repository import VadRepository
from app.shared.quality import METRIC_VERSION, SpeakerSide, SpeechSegment, TrackVadResult

VAD_VERSION = "naive-full-recording-v1"
VAD_FRAME_SECONDS = 0.03
VAD_MINIMUM_SPEECH_SECONDS = 0.12
VAD_MINIMUM_SILENCE_SECONDS = 0.4


def backfill_luel_vad(local_sessions_root: Path, minimum_quality: float) -> int:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for VAD backfill.")
    dataset_id = UUID("22647172-7eb6-4e72-97fd-a3e0b4885e3a")
    corpus_repository = CorpusAuditRepository(DATABASE_URL)
    sample_repository = Repository(DATABASE_URL)
    vad_repository = VadRepository(DATABASE_URL)
    evidence = corpus_repository.load_evidence(
        dataset_ids=(dataset_id,),
        minimum_quality=minimum_quality,
        metric_version=METRIC_VERSION,
        annotation_version=ANNOTATION_VERSION,
        region_analysis_version="conversation-regions-v1",
    )
    for index, item in enumerate(evidence, start=1):
        dashboard_sample = sample_repository.get_dashboard_sample(item.sample_id)
        for side in (TrackSide.SPEAKER1, TrackSide.SPEAKER2):
            track = _track(dashboard_sample, side)
            if track.audio_sha256 is None:
                raise ValueError(f"Missing audio hash for {track.id}")
            if track.duration_seconds is None:
                raise ValueError(f"Missing audio duration for {track.id}")
            if vad_repository.get_current(track.id, track.audio_sha256, VAD_VERSION) is not None:
                continue
            wave_path = (
                local_sessions_root / item.external_id / f"{item.external_id}_{side.value}.wav"
            )
            baseline = run_naive_vad_floor_full_recording(
                wave_path=wave_path,
                result_name=VAD_VERSION,
                description="Energy VAD backfill for materialized training metadata.",
                frame_seconds=VAD_FRAME_SECONDS,
                min_speech_seconds=VAD_MINIMUM_SPEECH_SECONDS,
                min_silence_seconds=VAD_MINIMUM_SILENCE_SECONDS,
            )
            vad_repository.upsert(
                sample_track_id=track.id,
                source_audio_sha256=track.audio_sha256,
                vad_version=VAD_VERSION,
                result=_track_vad(
                    side=side,
                    duration_seconds=track.duration_seconds,
                    baseline_segments=baseline.speech_segments,
                ),
            )
        print(f"[{index}/{len(evidence)}] {item.external_id}", flush=True)
    return len(evidence)


def _track(dashboard_sample: DashboardSample, side: TrackSide) -> SampleTrackRecord:
    for track in dashboard_sample.tracks:
        if track.side is side:
            return track
    raise ValueError(f"Missing {side.value} track.")


def _track_vad(
    side: TrackSide,
    duration_seconds: float,
    baseline_segments: list[BaselineSpeechSegment],
) -> TrackVadResult:
    segments = tuple(
        SpeechSegment(start_seconds=segment.start_seconds, end_seconds=segment.end_seconds)
        for segment in baseline_segments
    )
    durations = tuple(segment.duration_seconds for segment in segments)
    speech_seconds = sum(durations)
    return TrackVadResult(
        side=SpeakerSide.SPEAKER1 if side is TrackSide.SPEAKER1 else SpeakerSide.SPEAKER2,
        speech_segments=segments,
        speech_time_seconds=speech_seconds,
        speech_ratio=speech_seconds / duration_seconds,
        median_segment_duration_seconds=(statistics.median(durations) if durations else None),
        tiny_fragment_ratio=(
            sum(duration < 0.25 for duration in durations) / len(durations) if durations else 0.0
        ),
        long_segment_ratio=(
            sum(duration > 20.0 for duration in durations) / len(durations) if durations else 0.0
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-sessions-root", type=Path, required=True)
    parser.add_argument("--minimum-quality", type=float, default=0.95)
    options = parser.parse_args()
    count = backfill_luel_vad(
        local_sessions_root=options.local_sessions_root,
        minimum_quality=options.minimum_quality,
    )
    print(f"Backfilled VAD for {count} Luel recordings.")


if __name__ == "__main__":
    main()
