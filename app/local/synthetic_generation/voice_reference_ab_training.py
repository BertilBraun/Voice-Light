from __future__ import annotations

from pathlib import Path

from app.local.synthetic_generation.conversation_compiler import (
    AfterAssistantUserTiming,
    AfterUserAssistantTiming,
    CompiledConversation,
    CompletionPlacement,
    ConversationCompilerConfig,
    ConversationCompositionPlan,
    DuringAssistantUserTiming,
    FixedUserTiming,
    InterruptionFloorClaimPlacement,
    NonFloorFeedbackPlacement,
    RenderedUserClip,
    ResponseFloorClaimPlacement,
    VirtualAssistantTurn,
    compile_conversation,
)
from app.local.synthetic_generation.voice_reference_ab_pilot import (
    RenderedAuditionUtterance,
)


def compile_audition_candidate_training_samples(
    candidate_id: str,
    rendered_utterances: tuple[RenderedAuditionUtterance, ...],
    output_directory: Path,
    config: ConversationCompilerConfig,
) -> CompiledConversation:
    candidate_utterances = tuple(
        rendered for rendered in rendered_utterances if rendered.candidate_id == candidate_id
    )
    clips = tuple(_rendered_clip(rendered) for rendered in candidate_utterances)
    plan = _conversation_plan(candidate_id, clips)
    return compile_conversation(
        plan,
        clips,
        output_directory / "conversation.wav",
        config,
    )


def _rendered_clip(rendered: RenderedAuditionUtterance) -> RenderedUserClip:
    return RenderedUserClip(
        clip_id=rendered.utterance.utterance_id,
        audio_path=rendered.audio_path,
        audio_sha256=rendered.audio_sha256,
        duration_seconds=rendered.duration_seconds,
        active_start_seconds=0.0,
        active_end_seconds=rendered.duration_seconds,
        continuation_silences=rendered.continuation_silences,
    )


def _conversation_plan(
    candidate_id: str,
    clips: tuple[RenderedUserClip, ...],
) -> ConversationCompositionPlan:
    clips_by_id = {clip.clip_id: clip for clip in clips}
    expected_clip_ids = {"opening", "backchannel", "follow_up", "interruption", "closing"}
    if set(clips_by_id) != expected_clip_ids:
        raise ValueError(
            f"Candidate {candidate_id} must contain exactly the five audition utterances."
        )
    nominal_duration = sum(clip.duration_seconds for clip in clips) + 14.0
    return ConversationCompositionPlan(
        conversation_id=f"{candidate_id}_materialized",
        seed=37,
        duration_seconds=nominal_duration,
        user_events=(
            CompletionPlacement(
                event_id="opening",
                clip_id="opening",
                timing=FixedUserTiming(start_seconds=0.0),
            ),
            NonFloorFeedbackPlacement(
                event_id="backchannel",
                clip_id="backchannel",
                timing=DuringAssistantUserTiming(
                    assistant_turn_id="assistant_one",
                    position_fraction=0.5,
                ),
            ),
            ResponseFloorClaimPlacement(
                event_id="follow_up",
                clip_id="follow_up",
                timing=AfterAssistantUserTiming(
                    assistant_turn_id="assistant_one",
                    delay_seconds=0.24,
                ),
            ),
            InterruptionFloorClaimPlacement(
                event_id="interruption",
                clip_id="interruption",
                timing=DuringAssistantUserTiming(
                    assistant_turn_id="assistant_two",
                    position_fraction=0.75,
                ),
                assistant_yield_delay_seconds=0.24,
            ),
            ResponseFloorClaimPlacement(
                event_id="closing",
                clip_id="closing",
                timing=AfterAssistantUserTiming(
                    assistant_turn_id="assistant_three",
                    delay_seconds=0.24,
                ),
            ),
        ),
        assistant_turns=(
            VirtualAssistantTurn(
                turn_id="assistant_one",
                timing=AfterUserAssistantTiming(
                    user_event_id="opening",
                    delay_seconds=0.24,
                ),
                duration_seconds=4.0,
            ),
            VirtualAssistantTurn(
                turn_id="assistant_two",
                timing=AfterUserAssistantTiming(
                    user_event_id="follow_up",
                    delay_seconds=0.24,
                ),
                duration_seconds=3.0,
            ),
            VirtualAssistantTurn(
                turn_id="assistant_three",
                timing=AfterUserAssistantTiming(
                    user_event_id="interruption",
                    delay_seconds=0.24,
                ),
                duration_seconds=1.5,
            ),
        ),
    )
