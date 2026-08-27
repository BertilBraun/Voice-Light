from __future__ import annotations

from pathlib import Path

import numpy as np

from app.local.synthetic_generation.conversation_voice_references import (
    TtsBackendIdentity,
    file_sha256,
    write_pcm16_wave,
)
from app.local.synthetic_generation.voice_reference_ab_pilot import (
    REFERENCE_TEXT,
    ReferenceOrigin,
    RenderedAuditionUtterance,
    VoiceReferenceAbManifest,
    VoiceReferenceAbRenderManifest,
    VoiceReferenceCandidate,
    audition_utterances,
    compose_candidate_conversation,
    materialize_trimmed_audition_utterance,
    minimal_voice_design_candidates,
    render_voice_reference_ab_review,
)


def test_minimal_voice_design_matrix_has_two_unfiltered_candidates_per_identity() -> None:
    candidates = minimal_voice_design_candidates()

    assert len(candidates) == 8
    identities = {identity_id for _, identity_id, _, _ in candidates}
    assert len(identities) == 4
    assert all(
        sum(candidate_identity == identity_id for _, candidate_identity, _, _ in candidates) == 2
        for identity_id in identities
    )
    assert "clearly" not in REFERENCE_TEXT.casefold()
    assert "steadily" not in REFERENCE_TEXT.casefold()


def test_composes_and_renders_complete_reference_ab_review(tmp_path: Path) -> None:
    sample_rate_hz = 8_000
    references_directory = tmp_path / "references"
    renders_directory = tmp_path / "cosyvoice"
    reference_audio_path = references_directory / "audio" / "woman_easy_1.wav"
    utterance_audio_path = renders_directory / "audio" / "woman_easy_1_opening.wav"
    samples = np.full(sample_rate_hz, 0.1, dtype=np.float32)
    write_pcm16_wave(reference_audio_path, samples, sample_rate_hz)
    write_pcm16_wave(utterance_audio_path, samples, sample_rate_hz)
    backend = TtsBackendIdentity(
        backend_id="test",
        model_id="test/model",
        model_revision="revision",
        runtime_version="runtime",
        model_license="test-only",
    )
    candidate = VoiceReferenceCandidate(
        candidate_id="woman_easy_1",
        identity_id="woman_easy",
        candidate_index=1,
        identity_description="Adult woman, easy natural conversational voice.",
        origin=ReferenceOrigin.MINIMAL_VOICE_DESIGN,
        reference_text=REFERENCE_TEXT,
        voice_instruction="Adult woman, easy natural conversational voice.",
        request_seed=11,
        audio_path=Path("audio/woman_easy_1.wav"),
        audio_sha256=file_sha256(reference_audio_path),
        sample_rate_hz=sample_rate_hz,
        duration_seconds=1.0,
        generation_seconds=0.1,
    )
    references = VoiceReferenceAbManifest(backend=backend, candidates=(candidate,))
    references_path = references_directory / "references.json"
    references_path.write_text(references.model_dump_json(indent=2), encoding="utf-8")
    utterance = audition_utterances()[0].model_copy(update={"assistant_gap_after_seconds": 0.5})
    rendered_absolute = RenderedAuditionUtterance(
        candidate_id=candidate.candidate_id,
        utterance=utterance,
        audio_path=utterance_audio_path,
        audio_sha256=file_sha256(utterance_audio_path),
        sample_rate_hz=sample_rate_hz,
        original_duration_seconds=1.0,
        duration_seconds=1.0,
        trimmed_leading_seconds=0.0,
        trimmed_trailing_seconds=0.0,
        generation_seconds=0.1,
    )
    conversation_absolute = compose_candidate_conversation(
        candidate_id=candidate.candidate_id,
        rendered_utterances=(rendered_absolute,),
        output_path=renders_directory / "conversations" / "woman_easy_1.wav",
    )
    assert conversation_absolute.duration_seconds == 1.5
    rendered = rendered_absolute.model_copy(
        update={"audio_path": Path("audio/woman_easy_1_opening.wav")}
    )
    conversation = conversation_absolute.model_copy(
        update={"audio_path": Path("conversations/woman_easy_1.wav")}
    )
    renders = VoiceReferenceAbRenderManifest(
        references_sha256=file_sha256(references_path),
        backend=backend,
        utterances=(utterance,),
        rendered_utterances=(rendered,),
        conversations=(conversation,),
    )
    renders_path = renders_directory / "render.json"
    renders_path.write_text(renders.model_dump_json(indent=2), encoding="utf-8")
    review_path = render_voice_reference_ab_review(
        references_path,
        renders_path,
        tmp_path / "review" / "index.html",
    )

    review = review_path.read_text(encoding="utf-8")
    assert "Nothing is filtered" in review
    assert "Identical complete CosyVoice conversation" in review
    assert "../references/audio/woman_easy_1.wav" in review
    assert "../cosyvoice/conversations/woman_easy_1.wav" in review


def test_materializes_trimmed_audition_audio_before_timeline_composition(
    tmp_path: Path,
) -> None:
    sample_rate_hz = 8_000
    leading = np.zeros(round(0.2 * sample_rate_hz), dtype=np.float32)
    first_phrase = np.full(round(0.4 * sample_rate_hz), 0.1, dtype=np.float32)
    internal_pause = np.zeros(round(0.6 * sample_rate_hz), dtype=np.float32)
    second_phrase = np.full(round(0.4 * sample_rate_hz), 0.1, dtype=np.float32)
    trailing = np.zeros(round(0.3 * sample_rate_hz), dtype=np.float32)
    samples = np.concatenate((leading, first_phrase, internal_pause, second_phrase, trailing))

    rendered = materialize_trimmed_audition_utterance(
        candidate_id="woman_easy_1",
        utterance=audition_utterances()[1],
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        generation_seconds=0.2,
        output_path=tmp_path / "trimmed.wav",
    )

    assert rendered.original_duration_seconds == 1.9
    assert rendered.trimmed_leading_seconds == 0.2
    assert rendered.trimmed_trailing_seconds == 0.3
    assert rendered.duration_seconds == 1.4
    assert len(rendered.continuation_silences) == 1
    assert rendered.continuation_silences[0].start_seconds == 0.4
    assert rendered.continuation_silences[0].end_seconds == 1.0
