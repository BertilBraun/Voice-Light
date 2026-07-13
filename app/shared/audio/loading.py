from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from app.shared.quality import AudioMetadata
from app.shared.storage.base import StorageBackend


@dataclass(frozen=True)
class AudioTrack:
    samples: NDArray[np.float32]
    metadata: AudioMetadata


def probe_local_audio_metadata(path: Path) -> AudioMetadata:
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"Audio stream not found: {path}")
        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        channels = stream.codec_context.channels
        if sample_rate is None or channels is None:
            raise ValueError(f"Audio stream metadata is incomplete: {path}")
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = container.duration / 1_000_000
        else:
            raise ValueError(f"Audio duration is unavailable: {path}")
        sample_count = stream.frames if stream.frames > 0 else round(duration_seconds * sample_rate)
    return AudioMetadata(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        sample_count=sample_count,
    )


def load_audio(storage: StorageBackend, path: str) -> AudioTrack:
    with storage.open(path) as source:
        with av.open(source) as container:
            if not container.streams.audio:
                raise ValueError(f"Audio stream not found: {path}")
            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate
            channels = stream.codec_context.channels
            if sample_rate is None or channels is None:
                raise ValueError(f"Audio stream metadata is incomplete: {path}")
            resampler = av.AudioResampler(
                format="fltp",
                layout=stream.codec_context.layout.name,
                rate=sample_rate,
            )
            sample_parts = [
                resampled_frame.to_ndarray().T
                for decoded_frame in container.decode(stream)
                for resampled_frame in resampler.resample(decoded_frame)
            ]
            sample_parts.extend(
                resampled_frame.to_ndarray().T for resampled_frame in resampler.resample(None)
            )
    samples = (
        np.concatenate(sample_parts).astype(np.float32, copy=False)
        if sample_parts
        else np.empty((0, channels), dtype=np.float32)
    )
    return AudioTrack(
        samples=samples,
        metadata=AudioMetadata(
            duration_seconds=len(samples) / sample_rate,
            sample_rate=sample_rate,
            channels=channels,
            sample_count=len(samples),
        ),
    )
