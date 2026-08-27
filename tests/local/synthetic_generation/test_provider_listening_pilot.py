from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.provider_listening_pilot import (
    ConversationTurn,
    ListeningConversation,
    ListeningProvider,
    ListeningUtterance,
    ProviderListeningPlan,
    RenderedListeningUtterance,
    cloning_provider_listening_plan,
    compose_listening_conversations,
    file_sha256,
    qwen_custom_listening_plan,
    read_pcm16_wave,
    write_pcm16_wave,
)


def test_qwen_plan_contains_two_samples_for_each_selected_voice() -> None:
    plan = qwen_custom_listening_plan()

    counts = {
        voice: sum(utterance.voice is voice for utterance in plan.utterances)
        for voice in {utterance.voice for utterance in plan.utterances}
    }

    assert len(counts) == 4
    assert set(counts.values()) == {2}


@pytest.mark.parametrize(
    "provider",
    [ListeningProvider.COSYVOICE3, ListeningProvider.INDEXTTS25],
)
def test_clone_plan_has_requested_listening_coverage(provider: ListeningProvider) -> None:
    plan = cloning_provider_listening_plan(provider)

    backchannels = tuple(
        utterance for utterance in plan.utterances if utterance.purpose.value == "backchannel"
    )

    assert len(backchannels) == 7
    assert len({utterance.text for utterance in backchannels}) == 5
    assert len(plan.conversations) == 2


def test_plan_rejects_unknown_conversation_utterance() -> None:
    source = cloning_provider_listening_plan(ListeningProvider.COSYVOICE3)

    with pytest.raises(ValidationError, match="unknown utterance"):
        ProviderListeningPlan(
            plan_id="invalid_plan",
            provider=ListeningProvider.COSYVOICE3,
            utterances=source.utterances,
            conversations=(
                ListeningConversation(
                    conversation_id="bad_binding",
                    title="Bad binding",
                    turns=(
                        ConversationTurn(utterance_id="missing", assistant_gap_after_seconds=1.0),
                        ConversationTurn(
                            utterance_id="turn_short_key", assistant_gap_after_seconds=0.0
                        ),
                    ),
                ),
            ),
        )


def test_composition_marks_silent_intervals_as_assistant_activity(tmp_path: Path) -> None:
    source = cloning_provider_listening_plan(ListeningProvider.COSYVOICE3)
    conversation = source.conversations[0]
    selected_ids = {turn.utterance_id for turn in conversation.turns}
    plan = source.model_copy(
        update={
            "utterances": tuple(
                utterance
                for utterance in source.utterances
                if utterance.utterance_id in selected_ids
            ),
            "conversations": (conversation,),
        }
    )
    rendered = tuple(_rendered(utterance, tmp_path) for utterance in plan.utterances)

    conversations = compose_listening_conversations(plan, rendered, tmp_path / "composed")

    assert len(conversations) == 1
    result = conversations[0]
    assert (
        result.timeline[0].assistant_end_seconds - result.timeline[0].assistant_start_seconds == 2.8
    )
    assert result.timeline[-1].assistant_end_seconds == result.timeline[-1].speech_end_seconds
    samples, sample_rate_hz = read_pcm16_wave(result.audio_path)
    assert sample_rate_hz == 1000
    assert samples.size == round(result.duration_seconds * sample_rate_hz)


def _rendered(utterance: ListeningUtterance, output_directory: Path) -> RenderedListeningUtterance:
    audio_path = output_directory / f"{utterance.utterance_id}.wav"
    write_pcm16_wave(audio_path, np.full(1000, 0.25, dtype=np.float32), 1000)
    return RenderedListeningUtterance(
        utterance=utterance,
        speaker_label=utterance.voice.value,
        audio_path=audio_path,
        audio_sha256=file_sha256(audio_path),
        sample_rate_hz=1000,
        duration_seconds=1.0,
        generation_seconds=0.5,
        real_time_factor=0.5,
    )
