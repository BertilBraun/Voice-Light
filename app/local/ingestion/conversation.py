from __future__ import annotations

from app.local.analyses.asr.crosstalk import (
    CrosstalkFilterConfig,
    CrosstalkFramePower,
    filter_crosstalk_words,
    frame_power,
)
from app.local.analyses.asr.merger import parakeet_canary_union_words
from app.local.analyses.asr.service import word_from_timestamped_word
from app.local.analyses.end_of_turn.conversation_scoring import (
    ConversationScoringConfig,
    ScoredConversation,
    score_conversation,
)
from app.local.analyses.end_of_turn.detectors.asr_two_speaker_annotation import (
    SPEECH_SEGMENT_GAP_SECONDS,
    merged_asr_turns,
)
from app.local.analyses.end_of_turn.detectors.two_speaker_annotation import (
    OTHER_SPEAKER,
    TARGET_SPEAKER,
    TranscriptTurn,
)
from app.local.analyses.end_of_turn.service import SpeechSegment
from app.local.asr.full_recording_models import FullRecordingAsrTranscriptBundle
from app.local.ingestion.alignment import SynchronizationAlignment, shift_words
from app.shared.audio import AudioTrack
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationPoint,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    ConversationAnnotation,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)
from app.shared.quality_analysis.utils import clamp, safe_ratio
from app.shared.quality_analysis.vad import (
    VadConfig,
    speech_mask_for_samples,
    speech_segments_for_mask,
)

ANNOTATION_VERSION = "asr-full-parakeet-canary-two-speaker-annotation-v3"
SCORING_CONFIG = ConversationScoringConfig()
CROSSTALK_FILTER_CONFIG = CrosstalkFilterConfig()


def analyze_conversation(
    speaker1_audio: AudioTrack,
    speaker2_audio: AudioTrack,
    transcripts: FullRecordingAsrTranscriptBundle,
    alignment: SynchronizationAlignment,
) -> ConversationAnnotation:
    if speaker1_audio.metadata.sample_rate != speaker2_audio.metadata.sample_rate:
        raise ValueError("Full conversation annotation requires matching sample rates.")
    if len(speaker1_audio.samples) != len(speaker2_audio.samples):
        raise ValueError("Full conversation annotation requires a shared audio timeline.")
    duration_seconds = speaker1_audio.metadata.duration_seconds
    speaker1_words = shift_words(
        words=parakeet_canary_union_words(
            parakeet_words=tuple(
                word_from_timestamped_word(word) for word in transcripts.parakeet.speaker1.words
            ),
            canary_words=tuple(
                word_from_timestamped_word(word) for word in transcripts.canary.speaker1.words
            ),
        ),
        shift_seconds=0.0,
        timeline_duration_seconds=duration_seconds,
    )
    speaker2_words = shift_words(
        words=parakeet_canary_union_words(
            parakeet_words=tuple(
                word_from_timestamped_word(word) for word in transcripts.parakeet.speaker2.words
            ),
            canary_words=tuple(
                word_from_timestamped_word(word) for word in transcripts.canary.speaker2.words
            ),
        ),
        shift_seconds=alignment.speaker2_shift_seconds,
        timeline_duration_seconds=duration_seconds,
    )
    speaker1_power, speaker2_power = crosstalk_frame_power(
        speaker1_audio=speaker1_audio,
        speaker2_audio=speaker2_audio,
    )
    filtered_speaker1_words = filter_crosstalk_words(
        words=speaker1_words,
        frame_power_pair=speaker1_power,
        config=CROSSTALK_FILTER_CONFIG,
    )
    filtered_speaker2_words = filter_crosstalk_words(
        words=speaker2_words,
        frame_power_pair=speaker2_power,
        config=CROSSTALK_FILTER_CONFIG,
    )
    turns = merged_asr_turns(
        speaker1_words=filtered_speaker1_words,
        speaker2_words=filtered_speaker2_words,
        turn_gap_seconds=SPEECH_SEGMENT_GAP_SECONDS,
    )
    speaker1_scored = score_conversation(
        turns=turns,
        audio_activity_segments=audio_activity_segments(speaker1_audio),
        target_speaker=TARGET_SPEAKER,
        other_speaker=OTHER_SPEAKER,
        analysis_end_seconds=duration_seconds,
        config=SCORING_CONFIG,
    )
    speaker2_scored = score_conversation(
        turns=turns,
        audio_activity_segments=audio_activity_segments(speaker2_audio),
        target_speaker=OTHER_SPEAKER,
        other_speaker=TARGET_SPEAKER,
        analysis_end_seconds=duration_seconds,
        config=SCORING_CONFIG,
    )
    return conversation_annotation(
        turns=turns,
        duration_seconds=duration_seconds,
        speaker1=speaker_annotation(SpeakerSide.SPEAKER1, speaker1_scored),
        speaker2=speaker_annotation(SpeakerSide.SPEAKER2, speaker2_scored),
    )


def audio_activity_segments(audio: AudioTrack) -> list[SpeechSegment]:
    mono_samples = audio.samples[:, 0]
    config = VadConfig(relative_dominance_db=0.0)
    mask = speech_mask_for_samples(mono_samples, audio.metadata.sample_rate, config)
    segments = speech_segments_for_mask(mask, config)
    return [
        SpeechSegment(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )
        for segment in segments
    ]


def crosstalk_frame_power(
    speaker1_audio: AudioTrack,
    speaker2_audio: AudioTrack,
) -> tuple[CrosstalkFramePower, CrosstalkFramePower]:
    frame_size = max(
        1,
        round(CROSSTALK_FILTER_CONFIG.frame_duration_seconds * speaker1_audio.metadata.sample_rate),
    )
    speaker1_power = frame_power(samples=speaker1_audio.samples[:, 0], frame_size=frame_size)
    speaker2_power = frame_power(samples=speaker2_audio.samples[:, 0], frame_size=frame_size)
    shared_frame_count = min(len(speaker1_power), len(speaker2_power))
    frame_duration_seconds = frame_size / speaker1_audio.metadata.sample_rate
    return (
        CrosstalkFramePower(
            target=speaker1_power[:shared_frame_count],
            other=speaker2_power[:shared_frame_count],
            frame_duration_seconds=frame_duration_seconds,
        ),
        CrosstalkFramePower(
            target=speaker2_power[:shared_frame_count],
            other=speaker1_power[:shared_frame_count],
            frame_duration_seconds=frame_duration_seconds,
        ),
    )


def speaker_annotation(
    side: SpeakerSide,
    scored: ScoredConversation,
) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=tuple(
            AnnotationSpan(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=None,
            )
            for segment in scored.speech_segments
        ),
        pauses=tuple(
            AnnotationSpan(
                start_seconds=span.start_seconds,
                end_seconds=span.end_seconds,
                text=None,
            )
            for span in scored.pause_spans
        ),
        backchannels=tuple(
            AnnotationSpan(
                start_seconds=span.start_seconds,
                end_seconds=span.end_seconds,
                text=span.text,
            )
            for span in scored.backchannel_spans
        ),
        turns=tuple(
            AnnotationPoint(
                time_seconds=event.time_seconds,
                confidence=None,
                text=None,
            )
            for event in scored.end_of_turn_events
        ),
        interruptions=tuple(
            AnnotationPoint(
                time_seconds=event.time_seconds,
                confidence=event.confidence,
                text=event.text,
            )
            for event in scored.interruption_events
        ),
        segment_targets=tuple(
            SegmentAnnotationTarget(
                start_seconds=target.start_seconds,
                end_seconds=target.end_seconds,
                text=target.text,
                evidence_source=AnnotationEvidenceSource(target.evidence_source.value),
                keep_playing_confidence=target.keep_playing_confidence,
                turn_confidence=target.turn_confidence,
                interruption_confidence=target.interruption_confidence,
            )
            for target in scored.segment_hypotheses
        ),
        connection_targets=tuple(
            ConnectionAnnotationTarget(
                earlier_end_seconds=target.earlier_end_seconds,
                later_start_seconds=target.later_start_seconds,
                gap_seconds=target.gap_seconds,
                pause_confidence=target.pause_confidence,
                merge_confidence=target.merge_confidence,
            )
            for target in scored.connection_hypotheses
        ),
        speech_duration_seconds=sum(
            segment.end_seconds - segment.start_seconds for segment in scored.speech_segments
        ),
        pause_duration_seconds=sum(span.duration_seconds for span in scored.pause_spans),
        backchannel_duration_seconds=sum(
            span.duration_seconds for span in scored.backchannel_spans
        ),
    )


def conversation_annotation(
    turns: list[TranscriptTurn],
    duration_seconds: float,
    speaker1: SpeakerConversationAnnotation,
    speaker2: SpeakerConversationAnnotation,
) -> ConversationAnnotation:
    speech_segment_count = len(speaker1.speech_segments) + len(speaker2.speech_segments)
    turn_count = len(speaker1.turns) + len(speaker2.turns)
    pause_count = len(speaker1.pauses) + len(speaker2.pauses)
    backchannel_count = len(speaker1.backchannels) + len(speaker2.backchannels)
    interruption_count = len(speaker1.interruptions) + len(speaker2.interruptions)
    turn_taking_count = sum(
        1
        for earlier, later in zip(turns, turns[1:], strict=False)
        if earlier.speaker != later.speaker
    )
    usable_event_count = turn_count + pause_count + backchannel_count + interruption_count
    interaction_count = turn_taking_count + backchannel_count + interruption_count
    events_per_hour = safe_ratio(usable_event_count * 3600.0, duration_seconds)
    total_speech_seconds = speaker1.speech_duration_seconds + speaker2.speech_duration_seconds
    speaker_balance_score = (
        1.0
        - safe_ratio(
            abs(speaker1.speech_duration_seconds - speaker2.speech_duration_seconds),
            total_speech_seconds,
        )
        if total_speech_seconds > 0.0
        else 0.0
    )
    speech_coverage_score = clamp(
        safe_ratio(total_speech_seconds, duration_seconds) / 0.45, 0.0, 1.0
    )
    event_density_score = clamp(events_per_hour / 180.0, 0.0, 1.0)
    quality_score = clamp(
        (speech_coverage_score + event_density_score + speaker_balance_score) / 3.0,
        0.0,
        1.0,
    )
    return ConversationAnnotation(
        annotation_version=ANNOTATION_VERSION,
        analyzed_duration_seconds=duration_seconds,
        speaker1=speaker1,
        speaker2=speaker2,
        speech_segment_count=speech_segment_count,
        turn_count=turn_count,
        turn_taking_count=turn_taking_count,
        interaction_count=interaction_count,
        pause_count=pause_count,
        backchannel_count=backchannel_count,
        interruption_count=interruption_count,
        usable_event_count=usable_event_count,
        events_per_hour=events_per_hour,
        speaker_balance_score=speaker_balance_score,
        quality_score=quality_score,
    )
