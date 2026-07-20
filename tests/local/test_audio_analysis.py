import io
import subprocess
import wave
from pathlib import Path

import pytest

from app.local.analyses.end_of_turn.base import EndOfTurnDetectorMode
from app.local.analyses.end_of_turn.detectors.naive_vad import (
    _end_of_turn_events,
    _speech_segments_from_flags,
)
from app.local.analyses.end_of_turn.registry import available_detectors
from app.local.analyses.end_of_turn.router import parse_selected_detector_modes
from app.local.analyses.end_of_turn.service import SpeechSegment
from app.local.data.sessions import list_sessions
from app.local.data.transcripts import read_transcript_turns
from app.local.main import delete_file, write_playback_wave_file
from app.shared.audio.wav import (
    ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
    capped_audio_wave_bytes,
    capped_wave_bytes,
    wave_window_bytes,
)
from app.shared.audio.waveform import full_waveform_envelope


@pytest.mark.parametrize(
    ("speech_flags", "expected_segments"),
    [
        ([False, True, True, False], [SpeechSegment(start_seconds=0.1, end_seconds=0.3)]),
        ([True, False, True], []),
        ([True, True, True], [SpeechSegment(start_seconds=0.0, end_seconds=0.3)]),
    ],
)
def test_speech_segments_from_flags(
    speech_flags: list[bool],
    expected_segments: list[SpeechSegment],
) -> None:
    assert (
        _speech_segments_from_flags(
            speech_flags=speech_flags,
            frame_seconds=0.1,
            min_speech_seconds=0.2,
        )
        == expected_segments
    )


def test_end_of_turn_events_require_minimum_silence() -> None:
    speech_segments = [
        SpeechSegment(start_seconds=0.0, end_seconds=1.0),
        SpeechSegment(start_seconds=1.5, end_seconds=2.0),
        SpeechSegment(start_seconds=3.0, end_seconds=4.0),
    ]

    events = _end_of_turn_events(
        speech_segments=speech_segments,
        min_silence_seconds=0.7,
    )

    assert len(events) == 1
    assert events[0].time_seconds == 2.7
    assert events[0].speech_start_seconds == 1.5
    assert events[0].speech_end_seconds == 2.0
    assert events[0].silence_seconds == 1.0


def test_sessions_folder_does_not_require_deleted_source_manifest() -> None:
    sessions_root = Path("data/luel/sessions")

    assert sessions_root.is_dir()
    assert not (sessions_root / "manifest.json").exists()


def test_session_listing_uses_session_directories() -> None:
    session_identifiers = {session_entry.identifier for session_entry in list_sessions()}

    assert "pmt_001" in session_identifiers


def test_available_detectors_include_vad_variants() -> None:
    detector_modes = {detector_info.mode.value for detector_info in available_detectors()}

    assert detector_modes == {
        "naive_vad_floor",
        "naive_vad_fast",
        "silero_vad",
        "livekit_v1_mini",
        "pipecat_smart_turn_v2",
        "pipecat_smart_turn_v3",
        "transcript_gap",
        "two_speaker_annotation",
        "asr_two_speaker_annotation",
        "asr_two_speaker_annotation_speaker2",
        "turnsense",
    }


@pytest.mark.parametrize(
    ("detectors", "expected_modes"),
    [
        (None, None),
        ("", []),
        (
            "naive_vad_floor,silero_vad",
            [EndOfTurnDetectorMode.NAIVE_VAD_FLOOR, EndOfTurnDetectorMode.SILERO_VAD],
        ),
    ],
)
def test_parse_selected_detector_modes(
    detectors: str | None,
    expected_modes: list[EndOfTurnDetectorMode] | None,
) -> None:
    assert parse_selected_detector_modes(detectors=detectors) == expected_modes


def test_parse_selected_detector_modes_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown end-of-turn detector mode: missing"):
        parse_selected_detector_modes(detectors="missing")


def test_read_transcript_turns_loads_session_speakers() -> None:
    transcript_turns = read_transcript_turns(
        metadata_path=Path("data/luel/sessions/pmt_001/pmt_001.json")
    )

    assert transcript_turns[0].speaker == "Speaker1"
    assert transcript_turns[0].text.startswith("So apparently")
    assert transcript_turns[0].start_seconds == 0.26
    assert transcript_turns[1].speaker == "Speaker2"


def test_read_transcript_turns_applies_physical_alignment_sidecar() -> None:
    transcript_turns = read_transcript_turns(
        metadata_path=Path("data/luel/sessions/pmt_284/pmt_284.json")
    )

    assert transcript_turns[0].speaker == "Speaker1"
    assert transcript_turns[0].start_seconds == pytest.approx(7.46)
    first_speaker2_turn = next(
        transcript_turn
        for transcript_turn in transcript_turns
        if transcript_turn.speaker == "Speaker2"
    )
    assert first_speaker2_turn.start_seconds == pytest.approx(1.04)


def test_capped_wave_bytes_serves_browser_seekable_pcm16() -> None:
    wave_bytes = capped_wave_bytes(Path("data/luel/sessions/pmt_001/pmt_001_speaker1.wav"))

    with wave.open(io.BytesIO(wave_bytes), "rb") as wave_reader:
        assert wave_reader.getnchannels() == 1
        assert wave_reader.getsampwidth() == 2
        assert wave_reader.getnframes() / wave_reader.getframerate() == pytest.approx(
            ANALYSIS_AUDIO_MAX_DURATION_SECONDS
        )


def test_wave_window_bytes_reads_requested_recording_region() -> None:
    wave_bytes = wave_window_bytes(
        wave_path=Path("data/luel/sessions/pmt_001/pmt_001_speaker1.wav"),
        start_seconds=60.0,
        maximum_duration_seconds=10.0,
    )

    with wave.open(io.BytesIO(wave_bytes), "rb") as wave_reader:
        assert wave_reader.getnchannels() == 1
        assert wave_reader.getsampwidth() == 2
        assert wave_reader.getnframes() / wave_reader.getframerate() == pytest.approx(10.0)


def test_flac_playback_and_waveform_use_ffmpeg(tmp_path: Path) -> None:
    wave_path = tmp_path / "source.wav"
    flac_path = tmp_path / "source.flac"
    with wave.open(str(wave_path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(16_000)
        wave_writer.writeframes(b"\x00\x00" * 16_000)
    subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wave_path),
            str(flac_path),
        ),
        check=True,
    )

    playback = capped_audio_wave_bytes(flac_path)
    envelope = full_waveform_envelope(flac_path, point_count=100)

    with wave.open(io.BytesIO(playback), "rb") as wave_reader:
        assert wave_reader.getnchannels() == 1
        assert wave_reader.getsampwidth() == 2
        assert wave_reader.getnframes() / wave_reader.getframerate() == pytest.approx(1.0)
    assert envelope.duration_seconds == pytest.approx(1.0)
    assert len(envelope.points) == 100


def test_write_playback_wave_file_creates_and_deletes_capped_audio_file() -> None:
    playback_wave_path = write_playback_wave_file(
        wave_path=Path("data/luel/sessions/pmt_001/pmt_001_speaker1.wav")
    )

    try:
        assert playback_wave_path.exists()
        with wave.open(str(playback_wave_path), "rb") as wave_reader:
            assert wave_reader.getnchannels() == 1
            assert wave_reader.getsampwidth() == 2
            assert wave_reader.getnframes() / wave_reader.getframerate() == pytest.approx(
                ANALYSIS_AUDIO_MAX_DURATION_SECONDS
            )
    finally:
        delete_file(path=playback_wave_path)

    assert not playback_wave_path.exists()
