from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from app.local.synthetic_generation.corpus import (
    SyntheticCorpusRequest,
    build_synthetic_corpus,
)
from app.local.synthetic_generation.models import (
    SpeakerRole,
    SpeakerSpecification,
    SpeechEvent,
    SyntheticConversationPlan,
    TurnBoundary,
)
from app.local.synthetic_generation.rendering import (
    RenderedEventSource,
    SyntheticRenderRequest,
    TtsProvenance,
)
from app.local.training_corpus.export import ExportManifest, MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.local.training_samples.service import INPUT_DURATION_SECONDS
from app.training.turn_taking.hub import validate_sample_contract


def test_build_synthetic_corpus_matches_hugging_face_contract(tmp_path: Path) -> None:
    source_path = tmp_path / "source.wav"
    _write_tone(source_path)
    request = SyntheticCorpusRequest(
        corpus_id="pilot",
        description="A small train-only synthetic pilot.",
        plans=(
            SyntheticRenderRequest(
                plan=_plan(),
                sources=(_source(source_path),),
            ),
        ),
    )

    build_manifest = build_synthetic_corpus(request, tmp_path / "corpus")

    export_manifest = ExportManifest.model_validate_json(
        (tmp_path / "corpus" / "corpus.json").read_text(encoding="utf-8")
    )
    assert len(build_manifest.items) == 1
    assert export_manifest.training_sample_count == 1
    assert {shard.split for shard in export_manifest.shards} == {TrainingCorpusSplit.TRAIN}
    rows = pq.read_table(tmp_path / "corpus" / export_manifest.shards[0].path).to_pylist()
    sample = MaterializedTrainingSample.model_validate(rows[0])
    validate_sample_contract(sample, export_manifest, TrainingCorpusSplit.TRAIN)
    assert (tmp_path / "corpus" / sample.user_audio_path).is_file()
    assert (tmp_path / "corpus" / sample.assistant_audio_path).is_file()


def _plan() -> SyntheticConversationPlan:
    return SyntheticConversationPlan(
        plan_id="single_turn",
        description="One completed user turn.",
        seed=4,
        duration_seconds=INPUT_DURATION_SECONDS,
        speakers=(
            SpeakerSpecification(
                speaker_id="speaker_1",
                role=SpeakerRole.USER,
                voice_id="user",
                language="en-US",
            ),
            SpeakerSpecification(
                speaker_id="speaker_2",
                role=SpeakerRole.ASSISTANT,
                voice_id="assistant",
                language="en-US",
            ),
        ),
        events=(
            SpeechEvent(
                event_id="user_turn",
                speaker_id="speaker_1",
                text="Are you free tomorrow?",
                start_seconds=0.5,
                end_seconds=2.0,
                boundary_after=TurnBoundary.COMPLETION,
            ),
        ),
    )


def _source(path: Path) -> RenderedEventSource:
    return RenderedEventSource(
        event_id="user_turn",
        audio_path=path,
        audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        provenance=TtsProvenance(
            backend_id="test",
            runtime_version="1",
            model_id="tone",
            model_revision="1",
            model_license="CC0-1.0",
            seed=4,
        ),
    )


def _write_tone(path: Path) -> None:
    sample_rate_hz = 16_000
    times = np.arange(sample_rate_hz, dtype=np.float32) / sample_rate_hz
    samples = (np.sin(2.0 * np.pi * 220.0 * times) * 0.1 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate_hz)
        audio_file.writeframes(samples.tobytes())
