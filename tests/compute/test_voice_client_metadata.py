from app.compute.voice.client_metadata import create_voice_client_metadata


def test_voice_client_metadata_hashes_forwarded_address_and_bounds_headers() -> None:
    metadata = create_voice_client_metadata(
        origin=" https://voice.bertil-braun.de ",
        user_agent="browser/1 " + "x" * 600,
        forwarded_for="203.0.113.8, 10.0.0.1",
        peer_host="127.0.0.1",
        hash_key="deployment-secret",
    )

    assert metadata.origin == "https://voice.bertil-braun.de"
    assert metadata.user_agent is not None
    assert len(metadata.user_agent) == 512
    assert metadata.client_address_hash == "450570b7713e203f"


def test_voice_client_metadata_uses_peer_address_without_forwarding_header() -> None:
    first = create_voice_client_metadata(None, None, None, "127.0.0.1", "secret")
    repeated = create_voice_client_metadata(None, None, None, "127.0.0.1", "secret")
    other = create_voice_client_metadata(None, None, None, "127.0.0.2", "secret")

    assert first.client_address_hash == repeated.client_address_hash
    assert first.client_address_hash != other.client_address_hash
    assert first.origin is None
    assert first.user_agent is None
