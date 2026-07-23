from __future__ import annotations

import struct
from pathlib import Path

import pytest

from app.local.alignment_migration.filesystem_transaction import AlignmentTransactionPaths
from app.local.alignment_migration.riff import PCM_SUBFORMAT_GUID


def riff_chunk(identifier: bytes, payload: bytes, padding_byte: bytes = b"\x00") -> bytes:
    assert len(identifier) == 4
    padding = padding_byte if len(payload) % 2 else b""
    return identifier + struct.pack("<I", len(payload)) + payload + padding


def pcm_format_payload(
    format_tag: int,
    channel_count: int = 1,
    sample_rate: int = 8_000,
    bits_per_sample: int = 16,
) -> bytes:
    block_alignment = channel_count * ((bits_per_sample + 7) // 8)
    basic = struct.pack(
        "<HHIIHH",
        format_tag,
        channel_count,
        sample_rate,
        sample_rate * block_alignment,
        block_alignment,
        bits_per_sample,
    )
    if format_tag == 1:
        return basic
    if format_tag == 0xFFFE:
        return (
            basic
            + struct.pack("<H", 22)
            + struct.pack("<H", bits_per_sample)
            + struct.pack("<I", 0)
            + PCM_SUBFORMAT_GUID
        )
    return basic


def wave_bytes(
    pcm_data: bytes,
    format_tag: int = 1,
    channel_count: int = 1,
    sample_rate: int = 8_000,
    bits_per_sample: int = 16,
    chunks_before_data: tuple[bytes, ...] = (),
    chunks_after_data: tuple[bytes, ...] = (),
) -> bytes:
    chunks = (
        riff_chunk(
            b"fmt ",
            pcm_format_payload(
                format_tag=format_tag,
                channel_count=channel_count,
                sample_rate=sample_rate,
                bits_per_sample=bits_per_sample,
            ),
        ),
        *chunks_before_data,
        riff_chunk(b"data", pcm_data),
        *chunks_after_data,
    )
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.fixture
def alignment_paths(tmp_path: Path) -> AlignmentTransactionPaths:
    sample_directory = tmp_path / "sample_001"
    sample_directory.mkdir()
    (sample_directory / "sample_001_speaker1.wav").write_bytes(
        wave_bytes(
            pcm_data=b"\x11\x12\x21\x22\x31\x32",
            format_tag=0xFFFE,
            chunks_before_data=(riff_chunk(b"JUNK", b"abc", padding_byte=b"\x7f"),),
        )
    )
    (sample_directory / "sample_001_speaker2.wav").write_bytes(
        wave_bytes(
            pcm_data=b"\x41\x42\x51\x52\x61\x62\x71\x72",
            chunks_after_data=(riff_chunk(b"LIST", b"metadata!"),),
        )
    )
    return AlignmentTransactionPaths.create(
        sample_directory=sample_directory,
        sample_external_id="sample_001",
        speaker1_filename="sample_001_speaker1.wav",
        speaker2_filename="sample_001_speaker2.wav",
    )
