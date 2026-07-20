from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from app.local.asr.service import RemoteAsrClient
from app.local.db.models import (
    SampleLanguageStatus,
    TrackLanguageAssessment,
    TrackSide,
)
from app.shared.asr import AsrModelId, AsrTranscriptResult, RemoteAsrUploadRequest
from app.shared.language import (
    LANGUAGE_ASSESSMENT_VERSION,
    LanguageProbeWindow,
    TrackLanguageStatus,
    active_probe_windows,
    prepare_language_probe_audio,
    track_language_status,
)


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
