from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import av
from pydantic import Field

from app.shared.audio import probe_local_audio_metadata
from app.shared.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import QUALITY_SAMPLE_RATE

ASR_OPUS_BITRATE_BITS_PER_SECOND = 32_000
TRANSPORT_HASH_CHUNK_BYTES = 1024 * 1024
TRANSPORT_DURATION_TOLERANCE_SECONDS = 0.1


class AudioTransportCodec(StrEnum):
    FLAC = "flac"
    OGG_OPUS = "ogg_opus"


class AudioTransportMetadata(FrozenBaseModel):
    original_filename: str
    encoded_filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    codec: AudioTransportCodec
    duration_seconds: float = Field(gt=0.0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    size_bytes: int = Field(gt=0)


@dataclass(frozen=True)
class AudioTransportSpec:
    codec: AudioTransportCodec
    sample_rate: int
    channels: int
    maximum_duration_seconds: float | None
    bitrate_bits_per_second: int | None


ASR_AUDIO_TRANSPORT_SPEC = AudioTransportSpec(
    codec=AudioTransportCodec.OGG_OPUS,
    sample_rate=QUALITY_SAMPLE_RATE,
    channels=1,
    maximum_duration_seconds=None,
    bitrate_bits_per_second=ASR_OPUS_BITRATE_BITS_PER_SECOND,
)
QUALITY_AUDIO_TRANSPORT_SPEC = AudioTransportSpec(
    codec=AudioTransportCodec.FLAC,
    sample_rate=QUALITY_SAMPLE_RATE,
    channels=1,
    maximum_duration_seconds=ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
    bitrate_bits_per_second=None,
)
CANONICAL_MODEL_AUDIO_SPEC = AudioTransportSpec(
    codec=AudioTransportCodec.FLAC,
    sample_rate=QUALITY_SAMPLE_RATE,
    channels=1,
    maximum_duration_seconds=None,
    bitrate_bits_per_second=None,
)


def prepare_analysis_audio(source_path: Path, output_path: Path) -> AudioTransportMetadata:
    return prepare_audio_transport(
        source_path=source_path,
        output_path=output_path,
        spec=QUALITY_AUDIO_TRANSPORT_SPEC,
    )


def prepare_audio_transport(
    source_path: Path,
    output_path: Path,
    spec: AudioTransportSpec,
) -> AudioTransportMetadata:
    if not source_path.is_file():
        raise ValueError(f"Audio transport source file not found: {source_path}")
    validate_transport_spec(spec)
    command = audio_transport_command(
        source_path=source_path,
        output_path=output_path,
        spec=spec,
    )
    subprocess.run(command, check=True)
    output_metadata = probe_local_audio_metadata(output_path)
    if output_metadata.channels != spec.channels:
        raise ValueError(
            f"Prepared audio transport has {output_metadata.channels} channels, "
            f"expected {spec.channels}."
        )
    if spec.codec is AudioTransportCodec.FLAC and output_metadata.sample_rate != spec.sample_rate:
        raise ValueError(
            f"Prepared FLAC sample rate is {output_metadata.sample_rate}, "
            f"expected {spec.sample_rate}."
        )
    return AudioTransportMetadata(
        original_filename=source_path.name,
        encoded_filename=output_path.name,
        sha256=sha256_file(output_path),
        codec=spec.codec,
        duration_seconds=output_metadata.duration_seconds,
        sample_rate=spec.sample_rate,
        channels=spec.channels,
        size_bytes=output_path.stat().st_size,
    )


def audio_transport_command(
    source_path: Path,
    output_path: Path,
    spec: AudioTransportSpec,
) -> tuple[str, ...]:
    validate_transport_spec(spec)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
    ]
    if spec.maximum_duration_seconds is not None:
        command.extend(("-t", str(spec.maximum_duration_seconds)))
    command.extend(
        (
            "-map",
            "0:a:0",
            "-ac",
            str(spec.channels),
            "-ar",
            str(spec.sample_rate),
        )
    )
    match spec.codec:
        case AudioTransportCodec.FLAC:
            command.extend(("-c:a", "flac", "-compression_level", "0"))
        case AudioTransportCodec.OGG_OPUS:
            assert spec.bitrate_bits_per_second is not None
            command.extend(
                (
                    "-c:a",
                    "libopus",
                    "-b:a",
                    str(spec.bitrate_bits_per_second),
                    "-vbr",
                    "on",
                    "-application",
                    "audio",
                )
            )
    command.append(
        str(output_path),
    )
    return tuple(command)


def validate_transport_spec(spec: AudioTransportSpec) -> None:
    if spec.sample_rate <= 0:
        raise ValueError("Audio transport sample rate must be positive.")
    if spec.channels <= 0:
        raise ValueError("Audio transport channel count must be positive.")
    if spec.maximum_duration_seconds is not None and spec.maximum_duration_seconds <= 0.0:
        raise ValueError("Audio transport maximum duration must be positive.")
    if spec.codec is AudioTransportCodec.OGG_OPUS and (
        spec.bitrate_bits_per_second is None or spec.bitrate_bits_per_second <= 0
    ):
        raise ValueError("Ogg Opus transport requires a positive bitrate.")
    if spec.codec is AudioTransportCodec.FLAC and spec.bitrate_bits_per_second is not None:
        raise ValueError("FLAC transport does not accept a bitrate.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(TRANSPORT_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def encoded_audio_codec(path: Path) -> AudioTransportCodec:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"Audio stream not found: {path}")
        codec_name = container.streams.audio[0].codec_context.name
    match codec_name:
        case "flac":
            return AudioTransportCodec.FLAC
        case "opus":
            return AudioTransportCodec.OGG_OPUS
        case _:
            raise ValueError(f"Unsupported encoded audio codec: {codec_name}")
