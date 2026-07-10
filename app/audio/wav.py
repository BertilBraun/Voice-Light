from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mono_samples(
    fragment: bytes,
    sample_width: int,
    channel_count: int,
) -> NDArray[np.float64]:
    if sample_width == 1:
        unsigned_samples = np.frombuffer(fragment, dtype=np.uint8).reshape(-1, channel_count)
        return (unsigned_samples[:, 0].astype(np.float64) - 128.0).copy()
    if sample_width == 2:
        return (
            np.frombuffer(fragment, dtype="<i2").reshape(-1, channel_count)[:, 0].astype(np.float64)
        )
    if sample_width == 3:
        sample_bytes = np.frombuffer(fragment, dtype=np.uint8).reshape(-1, channel_count, 3)
        first_channel = sample_bytes[:, 0, :].astype(np.uint32)
        unsigned_values = (
            first_channel[:, 0] | (first_channel[:, 1] << 8) | (first_channel[:, 2] << 16)
        )
        signed_values = unsigned_values.astype(np.int32)
        signed_values[signed_values >= 0x800000] -= 0x1000000
        return signed_values.astype(np.float64)
    if sample_width == 4:
        return (
            np.frombuffer(fragment, dtype="<i4").reshape(-1, channel_count)[:, 0].astype(np.float64)
        )
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")
