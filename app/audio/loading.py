from __future__ import annotations

from dataclasses import dataclass

import av
import numpy as np
from numpy.typing import NDArray

from app.quality.models import AudioMetadata
from app.storage.base import StorageBackend


@dataclass(frozen=True)
class AudioTrack:
    samples: NDArray[np.float32]
    metadata: AudioMetadata


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
