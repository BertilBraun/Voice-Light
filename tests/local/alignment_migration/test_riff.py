from __future__ import annotations

import struct
from pathlib import Path

import pytest

from app.local.alignment_migration.riff import (
    MAXIMUM_UINT32,
    inspect_pcm_wave,
    rewrite_pcm_wave,
)
from tests.local.alignment_migration.conftest import (
    pcm_format_payload,
    riff_chunk,
    wave_bytes,
)


def test_rewrite_preserves_pcm_and_every_non_data_chunk(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "output.wav"
    junk_chunk = riff_chunk(b"JUNK", b"odd", padding_byte=b"\x7f")
    list_chunk = riff_chunk(b"LIST", b"metadata")
    original_pcm = b"\x01\x02\x03\x04\x05\x06"
    source_path.write_bytes(
        wave_bytes(
            pcm_data=original_pcm,
            chunks_before_data=(junk_chunk,),
            chunks_after_data=(list_chunk,),
        )
    )

    rewritten = rewrite_pcm_wave(
        source_path=source_path,
        output_path=output_path,
        prepended_silence_frame_count=2,
        target_frame_count=6,
    )

    assert rewritten.frame_count == 6
    assert rewritten.prepended_silence_frame_count == 2
    assert rewritten.appended_silence_frame_count == 1
    assert _data_payload(output_path) == b"\x00" * 4 + original_pcm + b"\x00" * 2
    assert _non_data_chunks(output_path) == _non_data_chunks(source_path)


@pytest.mark.parametrize(
    ("bits_per_sample", "original_pcm", "silence_byte"),
    (
        (8, b"\x01\xfe", b"\x80"),
        (16, b"\x01\x02\xfe\xff", b"\x00"),
        (24, b"\x01\x02\x03\xfd\xfe\xff", b"\x00"),
        (32, b"\x01\x02\x03\x04\xfc\xfd\xfe\xff", b"\x00"),
    ),
)
def test_rewrite_preserves_raw_pcm_and_encodes_silence_for_each_bit_depth(
    tmp_path: Path,
    bits_per_sample: int,
    original_pcm: bytes,
    silence_byte: bytes,
) -> None:
    source_path = tmp_path / f"source-{bits_per_sample}.wav"
    output_path = tmp_path / f"output-{bits_per_sample}.wav"
    bytes_per_frame = bits_per_sample // 8
    source_path.write_bytes(
        wave_bytes(
            pcm_data=original_pcm,
            bits_per_sample=bits_per_sample,
        )
    )

    rewrite_pcm_wave(
        source_path=source_path,
        output_path=output_path,
        prepended_silence_frame_count=1,
        target_frame_count=4,
    )

    expected_silence_frame = silence_byte * bytes_per_frame
    assert _data_payload(output_path) == (
        expected_silence_frame + original_pcm + expected_silence_frame
    )


def test_rewrite_accepts_pcm_extensible_and_preserves_24_bit_frames(tmp_path: Path) -> None:
    source_path = tmp_path / "extensible.wav"
    output_path = tmp_path / "aligned.wav"
    original_pcm = b"\x01\x02\x03\x11\x12\x13"
    source_path.write_bytes(
        wave_bytes(
            pcm_data=original_pcm,
            format_tag=0xFFFE,
            bits_per_sample=24,
            chunks_before_data=(riff_chunk(b"LIST", b"encoder"),),
        )
    )

    rewritten = rewrite_pcm_wave(
        source_path=source_path,
        output_path=output_path,
        prepended_silence_frame_count=1,
        target_frame_count=4,
    )

    metadata = inspect_pcm_wave(output_path)
    assert metadata.format_tag == 0xFFFE
    assert metadata.bits_per_sample == 24
    assert rewritten.appended_silence_frame_count == 1
    assert _data_payload(output_path) == b"\x00" * 3 + original_pcm + b"\x00" * 3
    assert _non_data_chunks(output_path) == _non_data_chunks(source_path)


@pytest.mark.parametrize("format_tag", (2, 3, 6))
def test_rewrite_rejects_compressed_wav(tmp_path: Path, format_tag: int) -> None:
    source_path = tmp_path / "compressed.wav"
    source_path.write_bytes(
        _riff_with_chunks(
            (
                riff_chunk(b"fmt ", pcm_format_payload(format_tag=format_tag)),
                riff_chunk(b"data", b"\x00\x00"),
            )
        )
    )

    with pytest.raises(ValueError, match="Compressed WAV"):
        rewrite_pcm_wave(source_path, tmp_path / "output.wav", 0, 1)


def test_rewrite_rejects_non_pcm_extensible_subtype(tmp_path: Path) -> None:
    source_path = tmp_path / "float-extensible.wav"
    format_payload = bytearray(pcm_format_payload(format_tag=0xFFFE))
    format_payload[24] = 3
    source_path.write_bytes(
        _riff_with_chunks(
            (
                riff_chunk(b"fmt ", bytes(format_payload)),
                riff_chunk(b"data", b"\x00\x00"),
            )
        )
    )

    with pytest.raises(ValueError, match="subtype is not PCM"):
        inspect_pcm_wave(source_path)


def test_rewrite_rejects_rf64(tmp_path: Path) -> None:
    source_path = tmp_path / "source.rf64"
    source_path.write_bytes(b"RF64" + struct.pack("<I", 4) + b"WAVE")

    with pytest.raises(ValueError, match="RF64"):
        inspect_pcm_wave(source_path)


def test_rewrite_rejects_multiple_data_chunks(tmp_path: Path) -> None:
    source_path = tmp_path / "duplicate-data.wav"
    source_path.write_bytes(
        _riff_with_chunks(
            (
                riff_chunk(b"fmt ", pcm_format_payload(format_tag=1)),
                riff_chunk(b"data", b"\x00\x00"),
                riff_chunk(b"data", b"\x01\x01"),
            )
        )
    )

    with pytest.raises(ValueError, match="exactly one data chunk"):
        inspect_pcm_wave(source_path)


@pytest.mark.parametrize(
    "source_bytes, expected_message",
    (
        (b"RIFF\x04\x00\x00\x00WAVE", "fmt chunk"),
        (
            b"RIFF\x0c\x00\x00\x00WAVEdata\x08\x00\x00\x00",
            "extends beyond",
        ),
    ),
)
def test_rewrite_rejects_malformed_riff(
    tmp_path: Path,
    source_bytes: bytes,
    expected_message: str,
) -> None:
    source_path = tmp_path / "malformed.wav"
    source_path.write_bytes(source_bytes)

    with pytest.raises(ValueError, match=expected_message):
        inspect_pcm_wave(source_path)


def test_rewrite_rejects_partial_pcm_frame(tmp_path: Path) -> None:
    source_path = tmp_path / "partial-frame.wav"
    source_path.write_bytes(wave_bytes(pcm_data=b"\x00\x01\x02"))

    with pytest.raises(ValueError, match="complete PCM frames"):
        inspect_pcm_wave(source_path)


def test_rewrite_rejects_unsupported_pcm_bit_depth(tmp_path: Path) -> None:
    source_path = tmp_path / "unsupported-depth.wav"
    source_path.write_bytes(
        wave_bytes(
            pcm_data=b"\x00\x00",
            bits_per_sample=12,
        )
    )

    with pytest.raises(ValueError, match="Unsupported WAV PCM bit depth"):
        inspect_pcm_wave(source_path)


def test_rewrite_rejects_riff_size_overflow_before_creating_output(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "output.wav"
    source_path.write_bytes(wave_bytes(pcm_data=b"\x00\x00"))

    with pytest.raises(ValueError, match="chunk-size limit"):
        rewrite_pcm_wave(
            source_path=source_path,
            output_path=output_path,
            prepended_silence_frame_count=0,
            target_frame_count=MAXIMUM_UINT32,
        )

    assert not output_path.exists()


def _data_payload(path: Path) -> bytes:
    metadata = inspect_pcm_wave(path)
    with path.open("rb") as source:
        source.seek(metadata.data_chunk.payload_offset)
        return source.read(metadata.data_chunk.payload_size)


def _non_data_chunks(path: Path) -> tuple[bytes, ...]:
    metadata = inspect_pcm_wave(path)
    chunks: list[bytes] = []
    with path.open("rb") as source:
        for chunk in metadata.chunks:
            if chunk.identifier == b"data":
                continue
            source.seek(chunk.header_offset)
            chunks.append(source.read(chunk.total_size))
    return tuple(chunks)


def _riff_with_chunks(chunks: tuple[bytes, ...]) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body
