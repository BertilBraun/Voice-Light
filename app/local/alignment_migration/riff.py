from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

RIFF_HEADER_SIZE = 12
CHUNK_HEADER_SIZE = 8
FORMAT_PCM = 1
FORMAT_EXTENSIBLE = 0xFFFE
PCM_SUBFORMAT_GUID = bytes.fromhex("0100000000001000800000aa00389b71")
SUPPORTED_PCM_BITS_PER_SAMPLE = frozenset({8, 16, 24, 32})
MAXIMUM_UINT32 = (1 << 32) - 1
COPY_BLOCK_SIZE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class RiffChunk:
    identifier: bytes
    header_offset: int
    payload_offset: int
    payload_size: int

    @property
    def padded_payload_size(self) -> int:
        return self.payload_size + (self.payload_size % 2)

    @property
    def total_size(self) -> int:
        return CHUNK_HEADER_SIZE + self.padded_payload_size


@dataclass(frozen=True)
class PcmWaveMetadata:
    path: Path
    file_size: int
    format_tag: int
    channel_count: int
    sample_rate: int
    byte_rate: int
    block_alignment: int
    bits_per_sample: int
    frame_count: int
    chunks: tuple[RiffChunk, ...]
    data_chunk: RiffChunk


@dataclass(frozen=True)
class RewrittenWave:
    sha256: str
    frame_count: int
    prepended_silence_frame_count: int
    appended_silence_frame_count: int


@dataclass(frozen=True)
class WaveTimelineSegment:
    output_frame_count: int
    source_start_frame: int | None


def inspect_pcm_wave(path: Path) -> PcmWaveMetadata:
    file_size = path.stat().st_size
    with path.open("rb") as source:
        header = _read_exact(source, RIFF_HEADER_SIZE, "RIFF header")
        container_identifier = header[:4]
        if container_identifier == b"RF64":
            raise ValueError(f"RF64 audio is not supported: {path}")
        if container_identifier != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"Audio is not a RIFF/WAVE file: {path}")
        declared_file_size = struct.unpack_from("<I", header, 4)[0] + 8
        if declared_file_size != file_size:
            raise ValueError(
                f"RIFF size does not match file size for {path}: "
                f"{declared_file_size} != {file_size}"
            )
        chunks = _read_chunks(source=source, file_size=file_size, path=path)

    format_chunks = tuple(chunk for chunk in chunks if chunk.identifier == b"fmt ")
    data_chunks = tuple(chunk for chunk in chunks if chunk.identifier == b"data")
    if len(format_chunks) != 1:
        raise ValueError(f"Expected exactly one fmt chunk in {path}, found {len(format_chunks)}")
    if len(data_chunks) != 1:
        raise ValueError(f"Expected exactly one data chunk in {path}, found {len(data_chunks)}")

    with path.open("rb") as source:
        source.seek(format_chunks[0].payload_offset)
        format_payload = _read_exact(source, format_chunks[0].payload_size, "fmt payload")
    (
        format_tag,
        channel_count,
        sample_rate,
        byte_rate,
        block_alignment,
        bits_per_sample,
    ) = _validate_pcm_format(format_payload=format_payload, path=path)
    data_chunk = data_chunks[0]
    if data_chunk.payload_size % block_alignment != 0:
        raise ValueError(f"WAV data is not aligned to complete PCM frames: {path}")
    return PcmWaveMetadata(
        path=path,
        file_size=file_size,
        format_tag=format_tag,
        channel_count=channel_count,
        sample_rate=sample_rate,
        byte_rate=byte_rate,
        block_alignment=block_alignment,
        bits_per_sample=bits_per_sample,
        frame_count=data_chunk.payload_size // block_alignment,
        chunks=chunks,
        data_chunk=data_chunk,
    )


def rewrite_pcm_wave(
    source_path: Path,
    output_path: Path,
    prepended_silence_frame_count: int,
    target_frame_count: int,
) -> RewrittenWave:
    if prepended_silence_frame_count < 0:
        raise ValueError("Prepended silence frame count must be non-negative.")
    metadata = inspect_pcm_wave(source_path)
    minimum_frame_count = metadata.frame_count + prepended_silence_frame_count
    if target_frame_count < minimum_frame_count:
        raise ValueError(
            f"Target frame count {target_frame_count} would cut audio from {source_path}."
        )
    appended_silence_frame_count = target_frame_count - minimum_frame_count
    output_data_size = target_frame_count * metadata.block_alignment
    if output_data_size > MAXIMUM_UINT32:
        raise ValueError(f"Aligned WAV data exceeds the RIFF chunk-size limit: {source_path}")
    output_data_padded_size = output_data_size + (output_data_size % 2)
    output_file_size = (
        metadata.file_size - metadata.data_chunk.padded_payload_size + output_data_padded_size
    )
    if output_file_size - 8 > MAXIMUM_UINT32:
        raise ValueError(f"Aligned WAV exceeds the RIFF file-size limit: {source_path}")

    try:
        with source_path.open("rb") as source, output_path.open("xb") as output:
            output.write(b"RIFF")
            output.write(struct.pack("<I", output_file_size - 8))
            output.write(b"WAVE")
            for chunk in metadata.chunks:
                if chunk.identifier == b"data":
                    output.write(b"data")
                    output.write(struct.pack("<I", output_data_size))
                    _write_silence_bytes(
                        output,
                        prepended_silence_frame_count * metadata.block_alignment,
                        metadata.bits_per_sample,
                    )
                    source.seek(chunk.payload_offset)
                    _copy_exact(source, output, chunk.payload_size)
                    _write_silence_bytes(
                        output,
                        appended_silence_frame_count * metadata.block_alignment,
                        metadata.bits_per_sample,
                    )
                    if output_data_size % 2:
                        output.write(b"\x00")
                else:
                    source.seek(chunk.header_offset)
                    _copy_exact(source, output, chunk.total_size)
            output.flush()
            os.fsync(output.fileno())
        if output_path.stat().st_size != output_file_size:
            raise ValueError(f"Aligned WAV size is inconsistent after writing: {output_path}")
        rewritten_metadata = inspect_pcm_wave(output_path)
        if rewritten_metadata.frame_count != target_frame_count:
            raise ValueError(f"Aligned WAV frame count is inconsistent: {output_path}")
        return RewrittenWave(
            sha256=sha256_file(output_path),
            frame_count=target_frame_count,
            prepended_silence_frame_count=prepended_silence_frame_count,
            appended_silence_frame_count=appended_silence_frame_count,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def rewrite_pcm_wave_timeline(
    source_path: Path,
    output_path: Path,
    segments: tuple[WaveTimelineSegment, ...],
) -> RewrittenWave:
    metadata = inspect_pcm_wave(source_path)
    if not segments:
        raise ValueError("Timeline rewrite requires at least one segment.")
    target_frame_count = sum(segment.output_frame_count for segment in segments)
    if target_frame_count <= 0:
        raise ValueError("Timeline rewrite must produce audio frames.")
    for segment in segments:
        if segment.output_frame_count < 0:
            raise ValueError("Timeline segment length cannot be negative.")
        if segment.source_start_frame is None:
            continue
        if segment.source_start_frame < 0:
            raise ValueError("Timeline source start cannot be negative.")
        if segment.source_start_frame + segment.output_frame_count > metadata.frame_count:
            raise ValueError("Timeline segment exceeds the source audio.")

    output_data_size = target_frame_count * metadata.block_alignment
    if output_data_size > MAXIMUM_UINT32:
        raise ValueError(f"Aligned WAV data exceeds the RIFF chunk-size limit: {source_path}")
    output_data_padded_size = output_data_size + (output_data_size % 2)
    output_file_size = (
        metadata.file_size - metadata.data_chunk.padded_payload_size + output_data_padded_size
    )
    if output_file_size - 8 > MAXIMUM_UINT32:
        raise ValueError(f"Aligned WAV exceeds the RIFF file-size limit: {source_path}")

    try:
        with source_path.open("rb") as source, output_path.open("xb") as output:
            output.write(b"RIFF")
            output.write(struct.pack("<I", output_file_size - 8))
            output.write(b"WAVE")
            for chunk in metadata.chunks:
                if chunk.identifier == b"data":
                    output.write(b"data")
                    output.write(struct.pack("<I", output_data_size))
                    for segment in segments:
                        byte_count = segment.output_frame_count * metadata.block_alignment
                        if segment.source_start_frame is None:
                            _write_silence_bytes(output, byte_count, metadata.bits_per_sample)
                        else:
                            source.seek(
                                chunk.payload_offset
                                + segment.source_start_frame * metadata.block_alignment
                            )
                            _copy_exact(source, output, byte_count)
                    if output_data_size % 2:
                        output.write(b"\x00")
                else:
                    source.seek(chunk.header_offset)
                    _copy_exact(source, output, chunk.total_size)
            output.flush()
            os.fsync(output.fileno())
        rewritten_metadata = inspect_pcm_wave(output_path)
        if rewritten_metadata.frame_count != target_frame_count:
            raise ValueError(f"Aligned WAV frame count is inconsistent: {output_path}")
        silence_frames = sum(
            segment.output_frame_count for segment in segments if segment.source_start_frame is None
        )
        return RewrittenWave(
            sha256=sha256_file(output_path),
            frame_count=target_frame_count,
            prepended_silence_frame_count=0,
            appended_silence_frame_count=silence_frames,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(COPY_BLOCK_SIZE_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _read_chunks(source: BinaryIO, file_size: int, path: Path) -> tuple[RiffChunk, ...]:
    chunks: list[RiffChunk] = []
    position = RIFF_HEADER_SIZE
    while position < file_size:
        if file_size - position < CHUNK_HEADER_SIZE:
            raise ValueError(f"Truncated RIFF chunk header in {path}")
        source.seek(position)
        chunk_header = _read_exact(source, CHUNK_HEADER_SIZE, "RIFF chunk header")
        payload_size = struct.unpack_from("<I", chunk_header, 4)[0]
        payload_offset = position + CHUNK_HEADER_SIZE
        padded_payload_size = payload_size + (payload_size % 2)
        next_position = payload_offset + padded_payload_size
        if next_position > file_size:
            raise ValueError(f"RIFF chunk extends beyond the file boundary in {path}")
        chunks.append(
            RiffChunk(
                identifier=chunk_header[:4],
                header_offset=position,
                payload_offset=payload_offset,
                payload_size=payload_size,
            )
        )
        position = next_position
    if position != file_size:
        raise ValueError(f"RIFF chunks do not end at the file boundary in {path}")
    return tuple(chunks)


def _validate_pcm_format(
    format_payload: bytes,
    path: Path,
) -> tuple[int, int, int, int, int, int]:
    if len(format_payload) < 16:
        raise ValueError(f"WAV fmt chunk is too short: {path}")
    values = struct.unpack_from("<HHIIHH", format_payload)
    format_tag, channel_count, sample_rate, byte_rate, block_alignment, bits_per_sample = values
    if format_tag == FORMAT_EXTENSIBLE:
        if len(format_payload) < 40:
            raise ValueError(f"WAVE_FORMAT_EXTENSIBLE fmt chunk is too short: {path}")
        extension_size = struct.unpack_from("<H", format_payload, 16)[0]
        if extension_size < 22 or len(format_payload) < 18 + extension_size:
            raise ValueError(f"WAVE_FORMAT_EXTENSIBLE extension is malformed: {path}")
        if format_payload[24:40] != PCM_SUBFORMAT_GUID:
            raise ValueError(f"WAVE_FORMAT_EXTENSIBLE subtype is not PCM: {path}")
    elif format_tag != FORMAT_PCM:
        raise ValueError(f"Compressed WAV format {format_tag} is not supported: {path}")
    if channel_count <= 0 or sample_rate <= 0 or bits_per_sample <= 0:
        raise ValueError(f"WAV PCM format contains non-positive dimensions: {path}")
    if bits_per_sample not in SUPPORTED_PCM_BITS_PER_SAMPLE:
        raise ValueError(f"Unsupported WAV PCM bit depth {bits_per_sample}: {path}")
    expected_block_alignment = channel_count * ((bits_per_sample + 7) // 8)
    if block_alignment != expected_block_alignment:
        raise ValueError(f"WAV block alignment is inconsistent with its PCM format: {path}")
    if byte_rate != sample_rate * block_alignment:
        raise ValueError(f"WAV byte rate is inconsistent with its PCM format: {path}")
    return values


def _read_exact(source: BinaryIO, byte_count: int, label: str) -> bytes:
    value = source.read(byte_count)
    if len(value) != byte_count:
        raise ValueError(f"Unexpected end of file while reading {label}.")
    return value


def _copy_exact(source: BinaryIO, output: BinaryIO, byte_count: int) -> None:
    remaining = byte_count
    while remaining:
        block = source.read(min(remaining, COPY_BLOCK_SIZE_BYTES))
        if not block:
            raise ValueError("Unexpected end of file while copying RIFF data.")
        output.write(block)
        remaining -= len(block)


def _write_silence_bytes(
    output: BinaryIO,
    byte_count: int,
    bits_per_sample: int,
) -> None:
    silence_byte = b"\x80" if bits_per_sample == 8 else b"\x00"
    silence_block = silence_byte * min(byte_count, COPY_BLOCK_SIZE_BYTES)
    remaining = byte_count
    while remaining:
        write_count = min(remaining, len(silence_block))
        output.write(silence_block[:write_count])
        remaining -= write_count
