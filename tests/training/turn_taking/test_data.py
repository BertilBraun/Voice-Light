import wave
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from app.local.training_corpus.audio_staging import transcode_lossless_flac
from app.training.turn_taking.data import (
    build_assistant_speaking_input,
    build_frame_targets,
    load_audio_window,
)
from app.training.turn_taking.schema import (
    ActivitySpan,
    DecisionTarget,
    EventTargetDistribution,
    TurnTakingSample,
)


def test_frame_targets_are_sparse_soft_and_separately_weighted() -> None:
    sample = TurnTakingSample(
        sample_id="sample",
        conversation_id="conversation",
        target_speaker_id="target",
        target_audio_path=Path("target.wav"),
        annotation_reference_audio_path=None,
        sample_rate_hz=16_000,
        context_start_seconds=0.0,
        decision_start_seconds=0.0,
        decision_end_seconds=3.0,
        assistant_speech_spans=(
            ActivitySpan(start_seconds=0.4, end_seconds=0.8),
            ActivitySpan(start_seconds=1.5, end_seconds=2.0),
        ),
        decisions=tuple(
            _decision(
                time_seconds=frame_index / 10,
                yield_probability=(0.7 if frame_index == 15 else 0.3),
                reliability=(0.4 if frame_index == 15 else None if frame_index == 20 else 0.9),
            )
            for frame_index in range(10, 30)
        ),
        source_dataset="test",
        source_license="test",
    )

    targets = build_frame_targets(
        sample,
        frame_seconds=0.1,
        burn_in_seconds=1.0,
        unmeasured_reliability_weight=0.75,
    )

    assert not targets.primary_mask[5]
    assert targets.primary_mask[15]
    assert targets.yield_probability[15].item() == pytest.approx(0.7)
    assert targets.primary_weight[15].item() == pytest.approx(0.4)
    assert targets.primary_weight[20].item() == pytest.approx(0.75)
    assert torch.equal(targets.future_activity_mask[15], torch.tensor([True, True, False, True]))
    assert targets.event_targets.shape == (30, 5)
    assert targets.event_mask.shape == (30, 5)
    assert targets.future_activity.shape == (30, 4)

    assistant_speaking = build_assistant_speaking_input(sample, frame_seconds=0.1)
    assert assistant_speaking[4]
    assert not assistant_speaking[8]
    assert assistant_speaking[15]
    assert not assistant_speaking[20]


def test_frame_targets_reject_missing_dense_primary_labels() -> None:
    sample = TurnTakingSample(
        sample_id="incomplete",
        conversation_id="conversation",
        target_speaker_id="target",
        target_audio_path=Path("target.wav"),
        annotation_reference_audio_path=None,
        sample_rate_hz=16_000,
        context_start_seconds=0.0,
        decision_start_seconds=1.0,
        decision_end_seconds=2.0,
        assistant_speech_spans=(),
        decisions=(_decision(time_seconds=1.0, yield_probability=1.0, reliability=None),),
        source_dataset="test",
        source_license="test",
    )

    with pytest.raises(ValueError, match="missing dense HOLD/YIELD labels"):
        build_frame_targets(
            sample,
            frame_seconds=0.1,
            burn_in_seconds=1.0,
            unmeasured_reliability_weight=1.0,
        )


@pytest.mark.parametrize("source_sample_rate", (16_000, 48_000))
@pytest.mark.parametrize(("start_seconds", "end_seconds"), ((0.0, 1.25), (2.25, 5.75)))
def test_audio_window_seek_matches_complete_flac_decode(
    tmp_path: Path,
    source_sample_rate: int,
    start_seconds: float,
    end_seconds: float,
) -> None:
    wave_path = tmp_path / "source.wav"
    flac_path = tmp_path / "source.flac"
    write_wave(wave_path, sample_rate=source_sample_rate, duration_seconds=8.0)
    transcode_lossless_flac(wave_path, flac_path)
    complete = decode_complete_audio(flac_path, sample_rate=16_000)

    window = load_audio_window(
        path=flac_path,
        sample_rate_hz=16_000,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    start_sample = round(start_seconds * 16_000)
    end_sample = round(end_seconds * 16_000)
    assert torch.equal(window, complete[start_sample:end_sample])


@pytest.mark.parametrize(
    ("sample_rate", "start_seconds", "end_seconds"),
    ((0, 0.0, 1.0), (16_000, -1.0, 1.0), (16_000, 1.0, 1.0)),
)
def test_audio_window_rejects_invalid_request(
    tmp_path: Path,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="sample_rate|Audio window"):
        load_audio_window(tmp_path / "unused.flac", sample_rate, start_seconds, end_seconds)


def _decision(
    time_seconds: float, yield_probability: float, reliability: float | None
) -> DecisionTarget:
    return DecisionTarget(
        time_seconds=time_seconds,
        yield_probability=yield_probability,
        primary_reliability=reliability,
        event_distribution=EventTargetDistribution(
            turn_completion=yield_probability,
            continuation_pause=1.0 - yield_probability,
            backchannel=0.0,
            interruption=0.0,
            other=0.0,
        ),
        event_reliability=reliability,
        future_user_activity=(False, True, None, True),
    )


def write_wave(path: Path, sample_rate: int, duration_seconds: float) -> None:
    sample_count = round(sample_rate * duration_seconds)
    times = np.arange(sample_count, dtype=np.float64) / sample_rate
    waveform = 0.45 * np.sin(2.0 * np.pi * 317.0 * times)
    waveform += 0.2 * np.sin(2.0 * np.pi * 911.0 * times)
    pcm = np.round(waveform * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def decode_complete_audio(path: Path, sample_rate: int) -> torch.Tensor:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        parts = [
            frame.to_ndarray().reshape(-1)
            for decoded in container.decode(stream)
            for frame in resampler.resample(decoded)
        ]
        parts.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(None))
    return torch.from_numpy(np.concatenate(parts).astype(np.float32, copy=False))
