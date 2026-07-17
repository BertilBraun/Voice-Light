from __future__ import annotations

import numpy as np
import pytest

from app.local.synchronization_review.acoustic_evaluation import best_acoustic_offset


def test_acoustic_onset_correlation_recovers_delayed_speaker2_envelope() -> None:
    random_generator = np.random.default_rng(seed=42)
    speaker1 = random_generator.normal(size=500).astype(np.float64)
    speaker2 = np.zeros_like(speaker1)
    speaker2[7:] = speaker1[:-7]

    predicted_shift, peak_correlation, peak_margin = best_acoustic_offset(
        speaker1_envelope=speaker1,
        speaker2_envelope=speaker2,
        frame_duration_seconds=0.1,
        maximum_lag_seconds=2.0,
    )

    assert predicted_shift == pytest.approx(-0.7)
    assert peak_correlation > 0.95
    assert peak_margin > 0.5
