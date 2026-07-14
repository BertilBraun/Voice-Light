from __future__ import annotations

import tempfile
from pathlib import Path

from app.local.analyses.asr.models import AsrModelMode
from app.local.analyses.asr.service import analyze_asr
from app.local.analyses.end_of_turn.conversation_scoring import (
    ConversationScoringConfig,
    ScoredConversation,
    score_conversation,
)
from app.local.analyses.end_of_turn.detectors.asr_two_speaker_annotation import (
    SPEECH_SEGMENT_GAP_SECONDS,
    merged_asr_turns,
    remote_asr_client,
)
from app.local.analyses.end_of_turn.detectors.two_speaker_annotation import (
    OTHER_SPEAKER,
    TARGET_SPEAKER,
    TranscriptTurn,
)
from app.local.analyses.end_of_turn.service import SpeechSegment
from app.local.asr.repository import AsrTranscriptRepository
from app.local.asr.transcript import SpeakerTrack
from app.shared.audio import load_audio
from app.shared.audio.transport import prepare_analysis_audio
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
from app.shared.storage.local import LocalStorageBackend

ANNOTATION_VERSION = "asr-two-speaker-annotation-v1"
ASR_MODEL_MODE = AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED
SCORING_CONFIG = ConversationScoringConfig()


def analyze_conversation(
    sample_id: str,
    speaker1_path: Path,
    speaker2_path: Path,
    database_url: str,
) -> ConversationAnnotation:
    with tempfile.TemporaryDirectory(prefix=f"voice-light-{sample_id}-") as directory:
        prepared_directory = Path(directory)
        prepared_speaker1_path = prepared_directory / f"{sample_id}_speaker1.flac"
        prepared_speaker2_path = prepared_directory / f"{sample_id}_speaker2.flac"
        prepare_analysis_audio(speaker1_path, prepared_speaker1_path)
        prepare_analysis_audio(speaker2_path, prepared_speaker2_path)
        cache = AsrTranscriptRepository(database_url=database_url)
        speaker1_response = analyze_asr(
            session_id=sample_id,
            speaker_track=SpeakerTrack.SPEAKER1,
            selected_models=(ASR_MODEL_MODE,),
            reference_words=(),
            cache=cache,
            remote_client_factory=remote_asr_client,
            source_audio_path=prepared_speaker1_path,
            other_audio_path=prepared_speaker2_path,
            prepared_audio_path=prepared_speaker1_path,
        )
        speaker2_response = analyze_asr(
            session_id=sample_id,
            speaker_track=SpeakerTrack.SPEAKER2,
            selected_models=(ASR_MODEL_MODE,),
            reference_words=(),
            cache=cache,
            remote_client_factory=remote_asr_client,
            source_audio_path=prepared_speaker2_path,
            other_audio_path=prepared_speaker1_path,
            prepared_audio_path=prepared_speaker2_path,
        )
        if len(speaker1_response.runs) != 1 or len(speaker2_response.runs) != 1:
            raise ValueError("Conversation annotation requires one ASR result per speaker.")
        turns = merged_asr_turns(
            speaker1_words=speaker1_response.runs[0].transcription.words,
            speaker2_words=speaker2_response.runs[0].transcription.words,
            turn_gap_seconds=SPEECH_SEGMENT_GAP_SECONDS,
        )
        duration_seconds = min(
            speaker1_response.analyzed_duration_seconds,
            speaker2_response.analyzed_duration_seconds,
        )
        speaker1_activity = audio_activity_segments(prepared_speaker1_path)
        speaker2_activity = audio_activity_segments(prepared_speaker2_path)
        speaker1_scored = score_conversation(
            turns=turns,
            audio_activity_segments=speaker1_activity,
            target_speaker=TARGET_SPEAKER,
            other_speaker=OTHER_SPEAKER,
            analysis_end_seconds=duration_seconds,
            config=SCORING_CONFIG,
        )
        speaker2_scored = score_conversation(
            turns=turns,
            audio_activity_segments=speaker2_activity,
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


def audio_activity_segments(audio_path: Path) -> list[SpeechSegment]:
    audio = load_audio(LocalStorageBackend(), audio_path.as_posix())
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
