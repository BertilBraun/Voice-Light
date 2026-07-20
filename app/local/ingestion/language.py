from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from app.local.asr.service import RemoteAsrClient
from app.local.db.models import TrackSide
from app.shared.asr import AsrModelId, AsrTranscriptResult, RemoteAsrUploadRequest
from app.shared.audio import probe_local_audio_metadata
from app.shared.audio.transport import (
    ASR_AUDIO_TRANSPORT_SPEC,
    AudioTransportCodec,
    AudioTransportMetadata,
    sha256_file,
)
from app.shared.base_model import FrozenBaseModel

LANGUAGE_ASSESSMENT_VERSION = "whisper-active-windows-v1"
LANGUAGE_PROBE_CANDIDATE_COUNT = 9
LANGUAGE_PROBE_WINDOW_COUNT = 3
LANGUAGE_PROBE_WINDOW_SECONDS = 8.0
LANGUAGE_PROBE_SAMPLE_RATE = 16_000
MINIMUM_ACTIVE_WINDOW_RMS_DBFS = -55.0
MINIMUM_LANGUAGE_CONFIDENCE = 0.75
MINIMUM_TRANSCRIPT_WORD_COUNT = 4


class TrackLanguageStatus(StrEnum):
    ENGLISH = "english"
    NON_ENGLISH = "non_english"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


class SampleLanguageStatus(StrEnum):
    ENGLISH = "english"
    NON_ENGLISH = "non_english"
    INCONCLUSIVE = "inconclusive"


class LanguageProbeWindow(FrozenBaseModel):
    start_seconds: float
    duration_seconds: float
    rms_dbfs: float


class TrackLanguageAssessment(FrozenBaseModel):
    sample_track_id: UUID
    source_audio_sha256: str
    assessment_version: str
    status: TrackLanguageStatus
    language_code: str | None
    confidence: float | None
    transcript_word_count: int
    transcript_text: str
    probe_windows: tuple[LanguageProbeWindow, ...]
    error: str | None


@dataclass(frozen=True)
class LanguageTrack:
    sample_track_id: UUID
    side: TrackSide
    source_path: Path
    source_audio_sha256: str


@dataclass(frozen=True)
class SampleLanguageAssessment:
    status: SampleLanguageStatus
    speaker1: TrackLanguageAssessment
    speaker2: TrackLanguageAssessment

    @property
    def allows_full_analysis(self) -> bool:
        return self.status is not SampleLanguageStatus.NON_ENGLISH


@dataclass(frozen=True)
class CandidateWindow:
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class CandidateWindowEnergy:
    window: CandidateWindow
    rms_dbfs: float


class LanguageAssessmentStore(Protocol):
    def get_current_assessment(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        assessment_version: str,
    ) -> TrackLanguageAssessment | None: ...

    def upsert_assessment(
        self,
        assessment: TrackLanguageAssessment,
    ) -> TrackLanguageAssessment: ...


class LanguageProbeTranscriber(Protocol):
    def transcribe_probe(
        self,
        source_path: Path,
        windows: tuple[LanguageProbeWindow, ...],
    ) -> AsrTranscriptResult: ...


class LanguagePreflightService:
    def __init__(
        self,
        store: LanguageAssessmentStore,
        transcriber: LanguageProbeTranscriber,
    ) -> None:
        self.store = store
        self.transcriber = transcriber

    def assess_sample(
        self,
        speaker1: LanguageTrack,
        speaker2: LanguageTrack,
    ) -> SampleLanguageAssessment:
        speaker1_assessment = self.assess_track(speaker1)
        speaker2_assessment = self.assess_track(speaker2)
        for assessment in (speaker1_assessment, speaker2_assessment):
            if assessment.status is TrackLanguageStatus.FAILED:
                raise ValueError(
                    f"Language preflight failed for {assessment.sample_track_id}: "
                    f"{assessment.error}"
                )
        return SampleLanguageAssessment(
            status=sample_language_status(speaker1_assessment, speaker2_assessment),
            speaker1=speaker1_assessment,
            speaker2=speaker2_assessment,
        )

    def assess_track(self, track: LanguageTrack) -> TrackLanguageAssessment:
        cached = self.store.get_current_assessment(
            sample_track_id=track.sample_track_id,
            source_audio_sha256=track.source_audio_sha256,
            assessment_version=LANGUAGE_ASSESSMENT_VERSION,
        )
        if cached is not None:
            return cached
        windows = active_probe_windows(track.source_path)
        try:
            transcript = self.transcriber.transcribe_probe(
                source_path=track.source_path,
                windows=windows,
            )
            assessment = assessment_from_transcript(
                sample_track_id=track.sample_track_id,
                source_audio_sha256=track.source_audio_sha256,
                windows=windows,
                transcript=transcript,
            )
        except Exception as error:
            assessment = TrackLanguageAssessment(
                sample_track_id=track.sample_track_id,
                source_audio_sha256=track.source_audio_sha256,
                assessment_version=LANGUAGE_ASSESSMENT_VERSION,
                status=TrackLanguageStatus.FAILED,
                language_code=None,
                confidence=None,
                transcript_word_count=0,
                transcript_text="",
                probe_windows=windows,
                error=f"{type(error).__name__}: {error}",
            )
        return self.store.upsert_assessment(assessment)


class RemoteWhisperLanguageProbeTranscriber:
    def __init__(self, remote_client: RemoteAsrClient) -> None:
        self.remote_client = remote_client

    def transcribe_probe(
        self,
        source_path: Path,
        windows: tuple[LanguageProbeWindow, ...],
    ) -> AsrTranscriptResult:
        if not windows:
            raise ValueError("Language preflight requires at least one active probe window.")
        output_file = tempfile.NamedTemporaryFile(
            prefix=f"{source_path.stem}_language_probe_",
            suffix=".ogg",
            delete=False,
        )
        output_path = Path(output_file.name)
        output_file.close()
        try:
            transport_metadata = prepare_language_probe_audio(
                source_path=source_path,
                windows=windows,
                output_path=output_path,
            )
            response = self.remote_client.transcribe_upload(
                request=RemoteAsrUploadRequest(
                    audio=transport_metadata,
                    models=(AsrModelId.WHISPERX,),
                ),
                audio_path=output_path,
            )
        finally:
            output_path.unlink(missing_ok=True)
        if len(response.results) != 1:
            raise ValueError("Language preflight ASR response must contain one Whisper result.")
        transcript = response.results[0]
        if transcript.model_id is not AsrModelId.WHISPERX:
            raise ValueError("Language preflight ASR response did not contain Whisper.")
        return transcript


def active_probe_windows(source_path: Path) -> tuple[LanguageProbeWindow, ...]:
    duration_seconds = probe_local_audio_metadata(source_path).duration_seconds
    candidates = candidate_windows(
        duration_seconds=duration_seconds,
        window_seconds=LANGUAGE_PROBE_WINDOW_SECONDS,
        candidate_count=LANGUAGE_PROBE_CANDIDATE_COUNT,
    )
    samples = decode_audio_windows(
        source_path=source_path,
        windows=candidates,
        sample_rate=LANGUAGE_PROBE_SAMPLE_RATE,
    )
    energies = tuple(
        CandidateWindowEnergy(
            window=window,
            rms_dbfs=rms_dbfs(window_samples),
        )
        for window, window_samples in zip(candidates, samples, strict=True)
    )
    return select_active_probe_windows(
        energies=energies,
        maximum_window_count=LANGUAGE_PROBE_WINDOW_COUNT,
        minimum_rms_dbfs=MINIMUM_ACTIVE_WINDOW_RMS_DBFS,
    )


def candidate_windows(
    duration_seconds: float,
    window_seconds: float,
    candidate_count: int,
) -> tuple[CandidateWindow, ...]:
    if duration_seconds <= 0.0:
        raise ValueError("Language probe source duration must be positive.")
    if window_seconds <= 0.0:
        raise ValueError("Language probe window duration must be positive.")
    if candidate_count <= 0:
        raise ValueError("Language probe candidate count must be positive.")
    actual_window_seconds = min(duration_seconds, window_seconds)
    maximum_start_seconds = max(0.0, duration_seconds - actual_window_seconds)
    if candidate_count == 1 or maximum_start_seconds == 0.0:
        return (
            CandidateWindow(
                start_seconds=0.0,
                duration_seconds=actual_window_seconds,
            ),
        )
    return tuple(
        CandidateWindow(
            start_seconds=maximum_start_seconds * index / (candidate_count - 1),
            duration_seconds=actual_window_seconds,
        )
        for index in range(candidate_count)
    )


def select_active_probe_windows(
    energies: tuple[CandidateWindowEnergy, ...],
    maximum_window_count: int,
    minimum_rms_dbfs: float,
) -> tuple[LanguageProbeWindow, ...]:
    if maximum_window_count <= 0:
        raise ValueError("Maximum language probe window count must be positive.")
    active = tuple(energy for energy in energies if energy.rms_dbfs >= minimum_rms_dbfs)
    selected = sorted(
        active,
        key=lambda energy: (-energy.rms_dbfs, energy.window.start_seconds),
    )[:maximum_window_count]
    return tuple(
        LanguageProbeWindow(
            start_seconds=energy.window.start_seconds,
            duration_seconds=energy.window.duration_seconds,
            rms_dbfs=energy.rms_dbfs,
        )
        for energy in sorted(selected, key=lambda energy: energy.window.start_seconds)
    )


def decode_audio_windows(
    source_path: Path,
    windows: tuple[CandidateWindow, ...],
    sample_rate: int,
) -> tuple[NDArray[np.float32], ...]:
    if not windows:
        raise ValueError("Audio window decoding requires at least one window.")
    if sample_rate <= 0:
        raise ValueError("Audio window sample rate must be positive.")
    command = audio_windows_command(
        source_path=source_path,
        windows=windows,
        sample_rate=sample_rate,
        output_path=None,
    )
    completed = subprocess.run(command, check=True, capture_output=True)
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    expected_counts = tuple(round(window.duration_seconds * sample_rate) for window in windows)
    expected_total = sum(expected_counts)
    if len(samples) != expected_total:
        raise ValueError(
            f"Decoded language probe has {len(samples)} samples, expected {expected_total}."
        )
    offsets = np.cumsum((0, *expected_counts))
    return tuple(
        samples[offsets[index] : offsets[index + 1]].copy() for index in range(len(windows))
    )


def prepare_language_probe_audio(
    source_path: Path,
    windows: tuple[LanguageProbeWindow, ...],
    output_path: Path,
) -> AudioTransportMetadata:
    subprocess.run(
        audio_windows_command(
            source_path=source_path,
            windows=tuple(
                CandidateWindow(
                    start_seconds=window.start_seconds,
                    duration_seconds=window.duration_seconds,
                )
                for window in windows
            ),
            sample_rate=ASR_AUDIO_TRANSPORT_SPEC.sample_rate,
            output_path=output_path,
        ),
        check=True,
    )
    metadata = probe_local_audio_metadata(output_path)
    return AudioTransportMetadata(
        original_filename=source_path.name,
        encoded_filename=output_path.name,
        sha256=sha256_file(output_path),
        codec=AudioTransportCodec.OGG_OPUS,
        duration_seconds=metadata.duration_seconds,
        sample_rate=ASR_AUDIO_TRANSPORT_SPEC.sample_rate,
        channels=1,
        size_bytes=output_path.stat().st_size,
    )


def audio_windows_command(
    source_path: Path,
    windows: tuple[CandidateWindow, ...],
    sample_rate: int,
    output_path: Path | None,
) -> tuple[str, ...]:
    if not source_path.is_file():
        raise ValueError(f"Language probe source file not found: {source_path}")
    if not windows:
        raise ValueError("Language probe command requires at least one window.")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    for window in windows:
        command.extend(
            (
                "-ss",
                str(window.start_seconds),
                "-t",
                str(window.duration_seconds),
                "-i",
                str(source_path),
            )
        )
    filter_parts = tuple(
        (
            f"[{index}:a:0]aresample={sample_rate},"
            f"aformat=sample_fmts=flt:channel_layouts=mono,"
            f"apad=whole_dur={window.duration_seconds},"
            f"atrim=duration={window.duration_seconds},"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
        for index, window in enumerate(windows)
    )
    inputs = "".join(f"[a{index}]" for index in range(len(windows)))
    concat = f"{inputs}concat=n={len(windows)}:v=0:a=1[out]"
    command.extend(("-filter_complex", ";".join((*filter_parts, concat)), "-map", "[out]"))
    if output_path is None:
        command.extend(("-f", "f32le", "pipe:1"))
    else:
        command.extend(
            (
                "-c:a",
                "libopus",
                "-b:a",
                str(ASR_AUDIO_TRANSPORT_SPEC.bitrate_bits_per_second),
                "-vbr",
                "on",
                "-application",
                "audio",
                str(output_path),
            )
        )
    return tuple(command)


def rms_dbfs(samples: NDArray[np.float32]) -> float:
    if len(samples) == 0:
        return -math.inf
    mean_square = float(np.mean(np.square(samples, dtype=np.float64)))
    if mean_square <= 0.0:
        return -math.inf
    return 10.0 * math.log10(mean_square)


def assessment_from_transcript(
    sample_track_id: UUID,
    source_audio_sha256: str,
    windows: tuple[LanguageProbeWindow, ...],
    transcript: AsrTranscriptResult,
) -> TrackLanguageAssessment:
    if transcript.error is not None:
        return TrackLanguageAssessment(
            sample_track_id=sample_track_id,
            source_audio_sha256=source_audio_sha256,
            assessment_version=LANGUAGE_ASSESSMENT_VERSION,
            status=TrackLanguageStatus.FAILED,
            language_code=None,
            confidence=None,
            transcript_word_count=0,
            transcript_text="",
            probe_windows=windows,
            error=transcript.error,
        )
    estimate = transcript.language_estimate
    word_count = len(transcript.words)
    status = track_language_status(
        language_code=estimate.language_code if estimate is not None else None,
        confidence=estimate.confidence if estimate is not None else None,
        transcript_word_count=word_count,
    )
    return TrackLanguageAssessment(
        sample_track_id=sample_track_id,
        source_audio_sha256=source_audio_sha256,
        assessment_version=LANGUAGE_ASSESSMENT_VERSION,
        status=status,
        language_code=estimate.language_code if estimate is not None else None,
        confidence=estimate.confidence if estimate is not None else None,
        transcript_word_count=word_count,
        transcript_text=transcript.text,
        probe_windows=windows,
        error=None,
    )


def track_language_status(
    language_code: str | None,
    confidence: float | None,
    transcript_word_count: int,
) -> TrackLanguageStatus:
    if language_code is None or confidence is None:
        return TrackLanguageStatus.INCONCLUSIVE
    if confidence < MINIMUM_LANGUAGE_CONFIDENCE:
        return TrackLanguageStatus.INCONCLUSIVE
    if transcript_word_count < MINIMUM_TRANSCRIPT_WORD_COUNT:
        return TrackLanguageStatus.INCONCLUSIVE
    normalized_code = language_code.strip().lower().replace("_", "-")
    if normalized_code == "en" or normalized_code.startswith("en-"):
        return TrackLanguageStatus.ENGLISH
    return TrackLanguageStatus.NON_ENGLISH


def sample_language_status(
    speaker1: TrackLanguageAssessment,
    speaker2: TrackLanguageAssessment,
) -> SampleLanguageStatus:
    statuses = (speaker1.status, speaker2.status)
    if TrackLanguageStatus.FAILED in statuses:
        raise ValueError("Failed track language assessments cannot form a sample decision.")
    if TrackLanguageStatus.NON_ENGLISH in statuses:
        return SampleLanguageStatus.NON_ENGLISH
    if TrackLanguageStatus.INCONCLUSIVE in statuses:
        return SampleLanguageStatus.INCONCLUSIVE
    return SampleLanguageStatus.ENGLISH
