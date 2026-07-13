from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def frame_rms(samples: NDArray[np.float32], frame_size: int) -> NDArray[np.float32]:
    if len(samples) == 0:
        return np.array([], dtype=np.float32)
    frame_count = int(np.ceil(len(samples) / frame_size))
    padded_length = frame_count * frame_size
    padded_samples = np.pad(samples, (0, padded_length - len(samples)))
    frames = padded_samples.reshape(frame_count, frame_size)
    return np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)
