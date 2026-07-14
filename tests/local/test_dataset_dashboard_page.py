from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.local.main import app


@pytest.mark.parametrize(
    "metric_label",
    (
        "Interactions",
        "Turns",
        "Turn takings",
        "Pauses",
        "Backchannels",
        "Interruptions",
        "Useful events",
    ),
)
def test_dataset_dashboard_explains_annotation_metrics(metric_label: str) -> None:
    with TestClient(app) as client:
        script_response = client.get("/pages/datasets/app.js")

    assert script_response.status_code == 200
    assert f'["{metric_label}",' in script_response.text
    assert 'tooltip.role = "tooltip"' in script_response.text
