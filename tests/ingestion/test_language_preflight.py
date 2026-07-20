from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
from numpy.typing import NDArray

from app.local.db.models import TrackSide
from app.local.ingestion.language import (
    LANGUAGE_ASSESSMENT_VERSION,
    LanguagePreflightService,
    LanguageTrack,
    SampleLanguageStatus,
    TrackLanguageAssessment,
    assessment_from_transcript,
    sample_language_status,
)
from app.shared.asr import (
    AsrModelId,
    AsrTranscriptResult,
    LanguageEstimate,
    TimestampedWord,
)
from app.shared.language import (
    CandidateWindow,
    CandidateWindowEnergy,
    LanguageProbeWindow,
    TrackLanguageStatus,
    candidate_windows,
    decode_audio_windows,
    rms_dbfs,
    select_active_probe_windows,
    track_language_status,
)


@dataclass
class MemoryLanguageAssessmentStore:
    assessments: dict[tuple[UUID, str, str], TrackLanguageAssessment] = field(default_factory=dict)
    upsert_count: int = 0

    def get_current_assessment(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        assessment_version: str,
    ) -> TrackLanguageAssessment | None:
        return self.assessments.get((sample_track_id, source_audio_sha256, assessment_version))

    def upsert_assessment(
        self,
        assessment: TrackLanguageAssessment,
    ) -> TrackLanguageAssessment:
        self.upsert_count += 1
        self.assessments[
            (
                assessment.sample_track_id,
                assessment.source_audio_sha256,
                assessment.assessment_version,
            )
        ] = assessment
        return assessment


@dataclass
class FixedLanguageProbeTranscriber:
    transcripts: dict[Path, AsrTranscriptResult]
    calls: list[Path] = field(default_factory=list)

    def transcribe_probe(
        self,
        source_path: Path,
        windows: tuple[LanguageProbeWindow, ...],
    ) -> AsrTranscriptResult:
        assert windows
        self.calls.append(source_path)
        return self.transcripts[source_path]


def test_candidate_windows_are_distributed_across_the_recording() -> None:
    windows = candidate_windows(
        duration_seconds=100.0,
        window_seconds=8.0,
        candidate_count=3,
    )

    assert windows == (
        CandidateWindow(start_seconds=0.0, duration_seconds=8.0),
        CandidateWindow(start_seconds=46.0, duration_seconds=8.0),
        CandidateWindow(start_seconds=92.0, duration_seconds=8.0),
    )


def test_candidate_window_for_short_recording_uses_available_audio() -> None:
    assert candidate_windows(
        duration_seconds=3.0,
        window_seconds=8.0,
        candidate_count=9,
    ) == (CandidateWindow(start_seconds=0.0, duration_seconds=3.0),)


def test_active_probe_windows_exclude_silence_and_preserve_timeline_order() -> None:
    energies = (
        CandidateWindowEnergy(CandidateWindow(0.0, 8.0), -70.0),
        CandidateWindowEnergy(CandidateWindow(20.0, 8.0), -20.0),
        CandidateWindowEnergy(CandidateWindow(40.0, 8.0), -35.0),
        CandidateWindowEnergy(CandidateWindow(60.0, 8.0), -25.0),
    )

    selected = select_active_probe_windows(
        energies=energies,
        maximum_window_count=2,
        minimum_rms_dbfs=-55.0,
    )

    assert selected == (
        LanguageProbeWindow(
            start_seconds=20.0,
            duration_seconds=8.0,
            rms_dbfs=-20.0,
        ),
        LanguageProbeWindow(
            start_seconds=60.0,
            duration_seconds=8.0,
            rms_dbfs=-25.0,
        ),
    )


def test_decode_audio_windows_reads_only_requested_regions(tmp_path: Path) -> None:
    source_path = tmp_path / "regions.wav"
    sample_rate = 16_000
    first = np.zeros(sample_rate, dtype=np.float32)
    second = np.full(sample_rate, 0.5, dtype=np.float32)
    write_float_wave(source_path, np.concatenate((first, second)), sample_rate)

    decoded = decode_audio_windows(
        source_path=source_path,
        windows=(
            CandidateWindow(start_seconds=0.0, duration_seconds=1.0),
            CandidateWindow(start_seconds=1.0, duration_seconds=1.0),
        ),
        sample_rate=sample_rate,
    )

    assert len(decoded) == 2
    assert rms_dbfs(decoded[0]) == -math.inf
    assert rms_dbfs(decoded[1]) == pytest.approx(-6.02, abs=0.05)


@pytest.mark.parametrize(
    ("language_code", "confidence", "word_count", "expected"),
    (
        ("en", 0.95, 10, TrackLanguageStatus.ENGLISH),
        ("en-US", 0.95, 10, TrackLanguageStatus.ENGLISH),
        ("de", 0.95, 10, TrackLanguageStatus.NON_ENGLISH),
        ("en", 0.70, 10, TrackLanguageStatus.INCONCLUSIVE),
        ("de", 0.95, 2, TrackLanguageStatus.INCONCLUSIVE),
        (None, None, 10, TrackLanguageStatus.INCONCLUSIVE),
    ),
)
def test_track_language_status_requires_confident_transcribed_speech(
    language_code: str | None,
    confidence: float | None,
    word_count: int,
    expected: TrackLanguageStatus,
) -> None:
    assert track_language_status(language_code, confidence, word_count) is expected


def test_assessment_preserves_language_evidence() -> None:
    track_id = uuid4()
    windows = (LanguageProbeWindow(start_seconds=10.0, duration_seconds=8.0, rms_dbfs=-20.0),)
    transcript = language_transcript("de", 0.98, "eins zwei drei vier fünf")

    assessment = assessment_from_transcript(
        sample_track_id=track_id,
        source_audio_sha256="a" * 64,
        windows=windows,
        transcript=transcript,
    )

    assert assessment.status is TrackLanguageStatus.NON_ENGLISH
    assert assessment.language_code == "de"
    assert assessment.confidence == 0.98
    assert assessment.transcript_word_count == 5
    assert assessment.probe_windows == windows


def test_sample_is_non_english_when_either_track_is_non_english() -> None:
    english = track_assessment(TrackLanguageStatus.ENGLISH, "en")
    german = track_assessment(TrackLanguageStatus.NON_ENGLISH, "de")

    assert sample_language_status(english, german) is SampleLanguageStatus.NON_ENGLISH


def test_language_preflight_caches_track_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker1_path = tmp_path / "speaker1.wav"
    speaker2_path = tmp_path / "speaker2.wav"
    write_float_wave(speaker1_path, np.full(16_000, 0.25, dtype=np.float32), 16_000)
    write_float_wave(speaker2_path, np.full(16_000, 0.25, dtype=np.float32), 16_000)
    probe_window = (LanguageProbeWindow(start_seconds=0.0, duration_seconds=1.0, rms_dbfs=-12.0),)
    monkeypatch.setattr(
        "app.local.ingestion.language.active_probe_windows",
        lambda source_path: probe_window,
    )
    store = MemoryLanguageAssessmentStore()
    transcriber = FixedLanguageProbeTranscriber(
        transcripts={
            speaker1_path: language_transcript("en", 0.99, "one two three four"),
            speaker2_path: language_transcript("en", 0.98, "five six seven eight"),
        }
    )
    service = LanguagePreflightService(store=store, transcriber=transcriber)
    speaker1 = language_track(speaker1_path, TrackSide.SPEAKER1, "a")
    speaker2 = language_track(speaker2_path, TrackSide.SPEAKER2, "b")

    first = service.assess_sample(speaker1=speaker1, speaker2=speaker2)
    second = service.assess_sample(speaker1=speaker1, speaker2=speaker2)

    assert first.status is SampleLanguageStatus.ENGLISH
    assert second == first
    assert transcriber.calls == [speaker1_path, speaker2_path]
    assert store.upsert_count == 2


def language_track(path: Path, side: TrackSide, hash_character: str) -> LanguageTrack:
    return LanguageTrack(
        sample_track_id=uuid4(),
        side=side,
        source_path=path,
        source_audio_sha256=hash_character * 64,
    )


def language_transcript(
    language_code: str,
    confidence: float,
    text: str,
) -> AsrTranscriptResult:
    return AsrTranscriptResult(
        model_id=AsrModelId.WHISPERX,
        text=text,
        words=tuple(
            TimestampedWord(
                text=word,
                start_seconds=float(index),
                end_seconds=float(index + 1),
            )
            for index, word in enumerate(text.split())
        ),
        language_estimate=LanguageEstimate(
            language_code=language_code,
            confidence=confidence,
        ),
    )


def track_assessment(
    status: TrackLanguageStatus,
    language_code: str | None,
) -> TrackLanguageAssessment:
    return TrackLanguageAssessment(
        sample_track_id=uuid4(),
        source_audio_sha256="a" * 64,
        assessment_version=LANGUAGE_ASSESSMENT_VERSION,
        status=status,
        language_code=language_code,
        confidence=0.99 if language_code is not None else None,
        transcript_word_count=4,
        transcript_text="one two three four",
        probe_windows=(),
        error=None,
    )


def write_float_wave(
    path: Path,
    samples: NDArray[np.float32],
    sample_rate: int,
) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(pcm.tobytes())
