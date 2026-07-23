from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.local.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.local.analyses.end_of_turn.detectors.transcript_gap import (
    TranscriptGapDetector,
    transcript_gap_detector,
)
from app.local.analyses.end_of_turn.service import (
    BackchannelSpan,
    EndOfTurnEvent,
    PauseSpan,
    SpeechSegment,
)


def test_transcript_gap_detector_builds_turns_and_events_from_speaker1_words(
    tmp_path: Path,
) -> None:
    speaker1_path = tmp_path / "sample_326_speaker1.wav"
    metadata_path = tmp_path / "sample_326.json"
    _write_metadata(
        metadata_path=metadata_path,
        duration_seconds=12.0,
        speaker1_words=[
            _word(text="hello", start_seconds=0.2, end_seconds=0.4),
            _word(text="there", start_seconds=0.8, end_seconds=1.0),
            _word(text="next", start_seconds=2.4, end_seconds=3.0),
            _word(text="last", start_seconds=8.0, end_seconds=8.3),
        ],
        speaker2_words=[
            _word(text="ignored", start_seconds=4.0, end_seconds=4.2),
        ],
    )

    result = _detector().analyze(speaker1_path=speaker1_path)

    assert result.speech_segments == [
        SpeechSegment(start_seconds=0.2, end_seconds=3.0),
        SpeechSegment(start_seconds=8.0, end_seconds=8.3),
    ]
    assert result.pause_spans == [
        PauseSpan(start_seconds=1.0, end_seconds=2.4, duration_seconds=1.4)
    ]
    assert result.backchannel_spans == []
    assert result.end_of_turn_events == [
        EndOfTurnEvent(
            time_seconds=3.0,
            speech_start_seconds=0.2,
            speech_end_seconds=3.0,
            silence_seconds=5.0,
        ),
        EndOfTurnEvent(
            time_seconds=8.3,
            speech_start_seconds=8.0,
            speech_end_seconds=8.3,
            silence_seconds=3.7,
        ),
    ]


def test_transcript_gap_detector_clips_words_and_trailing_silence_to_analysis_cap(
    tmp_path: Path,
) -> None:
    speaker1_path = tmp_path / "sample_326_speaker1.wav"
    metadata_path = tmp_path / "sample_326.json"
    _write_metadata(
        metadata_path=metadata_path,
        duration_seconds=240.0,
        speaker1_words=[
            _word(text="inside", start_seconds=177.0, end_seconds=177.4),
            _word(text="clipped", start_seconds=179.8, end_seconds=180.4),
            _word(text="outside", start_seconds=181.0, end_seconds=181.4),
        ],
        speaker2_words=[],
    )

    result = _detector().analyze(speaker1_path=speaker1_path)

    assert result.speech_segments == [
        SpeechSegment(start_seconds=177.0, end_seconds=177.4),
        SpeechSegment(start_seconds=179.8, end_seconds=180.0),
    ]
    assert result.pause_spans == []
    assert result.backchannel_spans == []
    assert result.end_of_turn_events == [
        EndOfTurnEvent(
            time_seconds=177.4,
            speech_start_seconds=177.0,
            speech_end_seconds=177.4,
            silence_seconds=2.4,
        )
    ]


def test_transcript_gap_detector_marks_short_acknowledgements_as_backchannels(
    tmp_path: Path,
) -> None:
    speaker1_path = tmp_path / "sample_326_speaker1.wav"
    metadata_path = tmp_path / "sample_326.json"
    _write_metadata(
        metadata_path=metadata_path,
        duration_seconds=7.0,
        speaker1_words=[
            _word(text="hello", start_seconds=0.0, end_seconds=0.3),
            _word(text="yeah", start_seconds=2.0, end_seconds=2.25),
            _word(text="continuing", start_seconds=4.0, end_seconds=4.4),
        ],
        speaker2_words=[],
    )

    result = _detector().analyze(speaker1_path=speaker1_path)

    assert result.speech_segments == [
        SpeechSegment(start_seconds=0.0, end_seconds=0.3),
        SpeechSegment(start_seconds=4.0, end_seconds=4.4),
    ]
    assert result.backchannel_spans == [
        BackchannelSpan(
            start_seconds=2.0,
            end_seconds=2.25,
            duration_seconds=0.25,
            text="yeah",
        )
    ]
    assert result.pause_spans == []
    assert result.end_of_turn_events == [
        EndOfTurnEvent(
            time_seconds=0.3,
            speech_start_seconds=0.0,
            speech_end_seconds=0.3,
            silence_seconds=3.7,
        ),
        EndOfTurnEvent(
            time_seconds=4.4,
            speech_start_seconds=4.0,
            speech_end_seconds=4.4,
            silence_seconds=2.6,
        ),
    ]


def test_transcript_gap_detector_keeps_contentful_acknowledgement_turns_as_speech(
    tmp_path: Path,
) -> None:
    speaker1_path = tmp_path / "sample_326_speaker1.wav"
    metadata_path = tmp_path / "sample_326.json"
    _write_metadata(
        metadata_path=metadata_path,
        duration_seconds=5.0,
        speaker1_words=[
            _word(text="yeah", start_seconds=0.0, end_seconds=0.2),
            _word(text="actually", start_seconds=0.35, end_seconds=0.75),
        ],
        speaker2_words=[],
    )

    result = _detector().analyze(speaker1_path=speaker1_path)

    assert result.speech_segments == [SpeechSegment(start_seconds=0.0, end_seconds=0.75)]
    assert result.backchannel_spans == []


def test_transcript_gap_detector_raises_for_missing_metadata(tmp_path: Path) -> None:
    speaker1_path = tmp_path / "sample_326_speaker1.wav"

    with pytest.raises(ValueError, match="Missing transcript metadata"):
        _detector().analyze(speaker1_path=speaker1_path)


def test_transcript_gap_detector_raises_for_missing_speaker1_transcript(
    tmp_path: Path,
) -> None:
    speaker1_path = tmp_path / "sample_326_speaker1.wav"
    metadata_path = tmp_path / "sample_326.json"
    _write_metadata(
        metadata_path=metadata_path,
        duration_seconds=12.0,
        speaker1_words=[],
        speaker2_words=[
            _word(text="other", start_seconds=0.2, end_seconds=0.4),
        ],
    )

    with pytest.raises(ValueError, match="Missing speaker transcript for Speaker1"):
        _detector().analyze(speaker1_path=speaker1_path)


def test_transcript_gap_factory_uses_transcript_gap_mode_when_wired() -> None:
    try:
        detector = transcript_gap_detector()
    except ValueError as error:
        assert "EndOfTurnDetectorMode.TRANSCRIPT_GAP" in str(error)
        return

    assert detector.info.mode.value == "transcript_gap"


def _detector() -> TranscriptGapDetector:
    return TranscriptGapDetector(
        info=EndOfTurnDetectorInfo(
            mode=EndOfTurnDetectorMode.NAIVE_VAD_FAST,
            label="Transcript gap",
            description="Test transcript gap detector.",
        ),
        turn_gap_seconds=1.5,
        internal_pause_seconds=0.5,
    )


def _write_metadata(
    metadata_path: Path,
    duration_seconds: float,
    speaker1_words: list[dict[str, object]],
    speaker2_words: list[dict[str, object]],
) -> None:
    speaker_transcript = []
    if speaker1_words:
        speaker_transcript.append(
            {
                "speaker": "Speaker1",
                "text": "",
                "startTime": speaker1_words[0]["startTime"],
                "endTime": speaker1_words[-1]["endTime"],
                "words": speaker1_words,
            }
        )
    if speaker2_words:
        speaker_transcript.append(
            {
                "speaker": "Speaker2",
                "text": "",
                "startTime": speaker2_words[0]["startTime"],
                "endTime": speaker2_words[-1]["endTime"],
                "words": speaker2_words,
            }
        )

    metadata_path.write_text(
        json.dumps(
            {
                "name": "sample_326",
                "durationSeconds": duration_seconds,
                "speakerTranscript": speaker_transcript,
            }
        ),
        encoding="utf-8",
    )


def _word(
    text: str,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, object]:
    return {
        "text": text,
        "startTime": start_seconds,
        "endTime": end_seconds,
    }
