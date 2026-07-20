from __future__ import annotations

from pathlib import Path

from app.local.ingestion.manifest import discover_manifest_samples, read_meetings_manifest


def test_manifest_discovery_uses_sorted_s3_keys_and_ignores_credentials(tmp_path: Path) -> None:
    manifest_path = tmp_path / "meetings.json"
    manifest_path.write_text(
        """
        {
          "schema_version": 1,
          "connection": {
            "provider": "aws_s3",
            "bucket": "meetings",
            "region": "eu-central-1",
            "prefix": "samples/",
            "s3_uri": "s3://meetings/samples/",
            "credential_mode": "environment",
            "credentials": {
              "access_key_id": "must-not-be-modeled",
              "secret_access_key": "must-not-be-modeled",
              "session_token": null
            }
          },
          "sample_count": 1,
          "file_count": 2,
          "total_bytes": 32000,
          "total_duration_seconds": 10.0,
          "samples": [
            {
              "sample_path": "sample-one",
              "sample_s3_uri": "s3://meetings/samples/sample-one",
              "file_count": 2,
              "total_bytes": 32000,
              "duration_seconds": 10.0,
              "duration_delta_seconds": 0.0,
              "quality": null,
              "files": [
                {
                  "key": "samples/sample-one/z.wav",
                  "s3_uri": "s3://meetings/samples/sample-one/z.wav",
                  "size_bytes": 16000,
                  "etag": "z",
                  "last_modified": "2026-01-01T00:00:00Z",
                  "extension": ".wav",
                  "duration_seconds": 10.0,
                  "bytes_per_second": 1600.0
                },
                {
                  "key": "samples/sample-one/a.wav",
                  "s3_uri": "s3://meetings/samples/sample-one/a.wav",
                  "size_bytes": 16000,
                  "etag": "a",
                  "last_modified": "2026-01-01T00:00:00Z",
                  "extension": ".wav",
                  "duration_seconds": 10.0,
                  "bytes_per_second": 1600.0
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    manifest = read_meetings_manifest(manifest_path)
    samples = discover_manifest_samples(manifest)

    assert len(samples) == 1
    assert samples[0].speaker1.uri.endswith("/a.wav")
    assert samples[0].speaker2.uri.endswith("/z.wav")
    assert "credentials" not in manifest.connection.model_dump()
