from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.local.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.local.analyses.end_of_turn.detectors.turnsense import (
    TurnsenseDetector,
    TurnsenseLabel,
    TurnsensePrediction,
    turnsense_detector,
)
from app.local.analyses.end_of_turn.service import EndOfTurnEvent, SpeechSegment


def test_turnsense_detector_classifies_speaker1_candidate_boundaries(
    tmp_path: Path,
) -> None:
    speaker1_path = tmp_path / "pmt_326_speaker1.wav"
    metadata_path = tmp_path / "pmt_326.json"
    _write_metadata(
        metadata_path=metadata_path,
        duration_seconds=12.0,
        speaker1_words=[
            _word(text="hello", start_seconds=0.2, end_seconds=0.4),
            _word(text="about", start_seconds=0.8, end_seconds=1.0),
            _word(text="billing", start_seconds=2.0, end_seconds=2.3),
            _word(text="thanks", start_seconds=7.0, end_seconds=7.3),
        ],
        speaker2_words=[
            _word(text="ignored", start_seconds=4.0, end_seconds=4.2),
        ],
    )

    result = _detector(
        predictions={
            "hello": TurnsensePrediction(
                label=TurnsenseLabel.NON_EOU,
                eou_probability=0.1,
            ),
            "hello about": TurnsensePrediction(
                label=TurnsenseLabel.NON_EOU,
                eou_probability=0.2,
            ),
            "hello about billing": TurnsensePrediction(
                label=TurnsenseLabel.EOU,
                eou_probability=0.9,
            ),
            "thanks": TurnsensePrediction(
                label=TurnsenseLabel.EOU,
                eou_probability=0.8,
            ),
        }
    ).analyze(speaker1_path=speaker1_path)

    assert result.speech_segments == [
        SpeechSegment(start_seconds=0.2, end_seconds=2.3),
        SpeechSegment(start_seconds=7.0, end_seconds=7.3),
    ]
    assert result.end_of_turn_events == [
        EndOfTurnEvent(
            time_seconds=2.8,
            speech_start_seconds=0.2,
            speech_end_seconds=2.3,
            silence_seconds=4.7,
        ),
        EndOfTurnEvent(
            time_seconds=7.8,
            speech_start_seconds=7.0,
            speech_end_seconds=7.3,
            silence_seconds=4.7,
        ),
    ]


def test_turnsense_detector_keeps_final_unfinished_segment_without_event(
    tmp_path: Path,
) -> None:
    speaker1_path = tmp_path / "pmt_326_speaker1.wav"
    metadata_path = tmp_path / "pmt_326.json"
    _write_metadata(
        metadata_path=metadata_path,
        duration_seconds=10.0,
        speaker1_words=[
            _word(text="one", start_seconds=0.0, end_seconds=0.2),
            _word(text="two", start_seconds=1.0, end_seconds=1.2),
        ],
        speaker2_words=[],
    )

    result = _detector(
        predictions={
            "one": TurnsensePrediction(
                label=TurnsenseLabel.EOU,
                eou_probability=0.95,
            ),
            "two": TurnsensePrediction(
                label=TurnsenseLabel.NON_EOU,
                eou_probability=0.1,
            ),
        }
    ).analyze(speaker1_path=speaker1_path)

    assert result.speech_segments == [
        SpeechSegment(start_seconds=0.0, end_seconds=0.2),
        SpeechSegment(start_seconds=1.0, end_seconds=1.2),
    ]
    assert result.end_of_turn_events == [
        EndOfTurnEvent(
            time_seconds=0.7,
            speech_start_seconds=0.0,
            speech_end_seconds=0.2,
            silence_seconds=0.8,
        )
    ]


def test_turnsense_detector_clips_trailing_silence_to_analysis_cap(
    tmp_path: Path,
) -> None:
    speaker1_path = tmp_path / "pmt_326_speaker1.wav"
    metadata_path = tmp_path / "pmt_326.json"
    _write_metadata(
        metadata_path=metadata_path,
        duration_seconds=240.0,
        speaker1_words=[
            _word(text="inside", start_seconds=179.2, end_seconds=179.4),
            _word(text="clipped", start_seconds=179.8, end_seconds=180.4),
            _word(text="outside", start_seconds=181.0, end_seconds=181.2),
        ],
        speaker2_words=[],
    )

    result = _detector(
        predictions={
            "inside clipped": TurnsensePrediction(
                label=TurnsenseLabel.EOU,
                eou_probability=0.99,
            ),
        }
    ).analyze(speaker1_path=speaker1_path)

    assert result.speech_segments == [
        SpeechSegment(start_seconds=179.2, end_seconds=180.0),
    ]
    assert result.end_of_turn_events == []


def test_turnsense_factory_uses_turnsense_mode_when_wired() -> None:
    try:
        detector = turnsense_detector()
    except ValueError as error:
        assert "EndOfTurnDetectorMode.TURNSENSE" in str(error)
        return

    assert detector.info.mode.value == "turnsense"


@dataclass(frozen=True)
class _ScriptedClassifier:
    predictions: dict[str, TurnsensePrediction]

    def predict_end_of_turn(self, text: str) -> TurnsensePrediction:
        return self.predictions[text]


def _detector(
    predictions: dict[str, TurnsensePrediction],
) -> TurnsenseDetector:
    return TurnsenseDetector(
        info=EndOfTurnDetectorInfo(
            mode=EndOfTurnDetectorMode.NAIVE_VAD_FAST,
            label="Turnsense",
            description="Test Turnsense detector.",
        ),
        classifier=_ScriptedClassifier(predictions=predictions),
        min_silence_seconds=0.5,
        eou_threshold=0.5,
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
                "name": "pmt_326",
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
