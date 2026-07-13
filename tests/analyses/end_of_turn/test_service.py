from __future__ import annotations

from app.local.analyses.end_of_turn.service import (
    BackchannelSpan,
    BaselineResult,
    EndOfTurnEvent,
    PauseSpan,
    SpeechSegment,
    baseline_to_json,
)


def test_baseline_to_json_includes_pause_and_backchannel_spans() -> None:
    result = BaselineResult(
        name="transcript_gap",
        description="test",
        frame_seconds=0.5,
        min_silence_seconds=1.5,
        threshold=1.5,
        speech_segments=[SpeechSegment(start_seconds=0.0, end_seconds=2.0)],
        pause_spans=[PauseSpan(start_seconds=0.4, end_seconds=1.0, duration_seconds=0.6)],
        backchannel_spans=[
            BackchannelSpan(
                start_seconds=3.0,
                end_seconds=3.2,
                duration_seconds=0.2,
                text="yeah",
            )
        ],
        end_of_turn_events=[
            EndOfTurnEvent(
                time_seconds=2.0,
                speech_start_seconds=0.0,
                speech_end_seconds=2.0,
                silence_seconds=1.5,
            )
        ],
    )

    assert baseline_to_json(result) == {
        "name": "transcript_gap",
        "description": "test",
        "frame_seconds": 0.5,
        "min_silence_seconds": 1.5,
        "threshold": 1.5,
        "speech_segments": [{"start_seconds": 0.0, "end_seconds": 2.0}],
        "pause_spans": [{"start_seconds": 0.4, "end_seconds": 1.0, "duration_seconds": 0.6}],
        "backchannel_spans": [
            {
                "start_seconds": 3.0,
                "end_seconds": 3.2,
                "duration_seconds": 0.2,
                "text": "yeah",
            }
        ],
        "end_of_turn_events": [
            {
                "time_seconds": 2.0,
                "speech_start_seconds": 0.0,
                "speech_end_seconds": 2.0,
                "silence_seconds": 1.5,
            }
        ],
        "interruption_events": [],
    }
