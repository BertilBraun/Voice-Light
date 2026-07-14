from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.local.main import app


@pytest.fixture(scope="module")
def dashboard_script() -> Iterator[str]:
    with TestClient(app) as client:
        script_response = client.get("/pages/datasets/app.js")
    assert script_response.status_code == 200
    yield script_response.text


@pytest.mark.parametrize(
    "metric_label",
    (
        "Analyzed samples",
        "Invalid samples",
        "Analyzed audio",
        "Speech segments",
        "Interactions",
        "Turns",
        "Turn takings",
        "Pauses",
        "Backchannels",
        "Interruptions",
        "Useful events",
        "Overall",
        "Calibrated",
        "Raw",
        "Interaction",
        "Timing",
        "Audio",
        "Conversation",
        "Analyzed duration",
        "Events / hour",
        "Speaker balance",
        "Speech",
        "Silence",
        "Overlap",
        "Candidates / hour",
        "Turns / hour",
        "Responses / hour",
        "Interruptions / hour",
        "Backchannels / hour",
        "Median segment",
        "Median turn gap",
        "Median pause",
        "Median overlap",
        "Tiny fragments",
        "Long segments",
        "Duration gap",
        "Duration mismatch",
        "Track correlation",
        "Envelope correlation",
        "Speaker 1 leakage",
        "Speaker 2 leakage",
    ),
)
def test_dataset_dashboard_explains_metric(metric_label: str, dashboard_script: str) -> None:
    assert f'["{metric_label}", "' in dashboard_script
    assert 'tooltip.role = "tooltip"' in dashboard_script


@pytest.mark.parametrize(
    "track_metric_label",
    ("Score", "RMS", "Peak", "Clipping", "Near zero"),
)
def test_dataset_dashboard_explains_audio_track_metric(
    track_metric_label: str,
    dashboard_script: str,
) -> None:
    assert f'["{track_metric_label}", "' in dashboard_script
    assert "audioTrackMetricDescriptions.get(label)" in dashboard_script


@pytest.mark.parametrize(
    "event_type",
    (
        "turn_completion",
        "pause",
        "start_response",
        "interruption",
        "backchannel",
        "overlap",
    ),
)
def test_dataset_dashboard_explains_event_type(event_type: str, dashboard_script: str) -> None:
    assert f'["{event_type}", "' in dashboard_script
    assert "eventTypeDescriptions.get(eventType)" in dashboard_script
