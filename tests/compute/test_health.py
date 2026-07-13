from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.compute.config import ComputeSettings
from app.compute.main import create_compute_app


def test_liveness_is_public_and_readiness_requires_authentication(tmp_path: Path) -> None:
    application = create_compute_app(
        ComputeSettings(token="secret-token", log_directory=tmp_path / "logs")
    )
    client = TestClient(application)

    assert client.get("/health/live").json() == {"status": "live"}
    assert client.get("/health/ready").status_code == 401

    response = client.get(
        "/health/ready",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert [stage["status"] for stage in response.json()["stages"]] == [
        "pending",
        "pending",
        "pending",
    ]
    assert response.headers["X-Request-ID"]
