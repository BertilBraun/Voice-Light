from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceClientMetadata:
    origin: str | None
    user_agent: str | None
    client_address_hash: str | None


def create_voice_client_metadata(
    origin: str | None,
    user_agent: str | None,
    forwarded_for: str | None,
    peer_host: str | None,
    hash_key: str,
) -> VoiceClientMetadata:
    forwarded_host = forwarded_for.split(",", maxsplit=1)[0].strip() if forwarded_for else None
    client_host = forwarded_host or peer_host
    return VoiceClientMetadata(
        origin=_bounded_header(origin, maximum_length=256),
        user_agent=_bounded_header(user_agent, maximum_length=512),
        client_address_hash=_hash_client_address(client_host, hash_key),
    )


def _bounded_header(value: str | None, maximum_length: int) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized[:maximum_length] or None


def _hash_client_address(client_host: str | None, hash_key: str) -> str | None:
    if client_host is None:
        return None
    digest = hmac.new(
        hash_key.encode("utf-8"),
        client_host.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest[:16]
