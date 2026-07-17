from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from app.compute.voice.schemas import CausalSource


class OverlapResolutionKind(StrEnum):
    NON_FLOOR_TAKING = "non_floor_taking"
    RESPONSE_REQUIRED = "response_required"
    FLOOR_TAKING = "floor_taking"
    UNRESOLVED = "unresolved"


class OverlapResolutionReason(StrEnum):
    ACOUSTIC_BURST_ENDED = "acoustic_burst_ended"
    CLOSED_ACKNOWLEDGEMENT = "closed_acknowledgement"
    SHORT_LAUGHTER = "short_laughter"
    EXPLICIT_STOP_OR_REPAIR = "explicit_stop_or_repair"
    QUESTION = "question"
    MEANINGFUL_LEXICAL_MATERIAL = "meaningful_lexical_material"
    ACKNOWLEDGEMENT_CONTINUATION = "acknowledgement_continuation"
    PREDICTED_INTERRUPTION = "predicted_interruption"
    SPEECH_DURATION_DEADLINE = "speech_duration_deadline"
    AWAITING_MORE_EVIDENCE = "awaiting_more_evidence"


@dataclass(frozen=True)
class ProvisionalOverlapPolicyConfig:
    acknowledgement_phrases: tuple[str, ...] = (
        "mm hm",
        "mhm",
        "uh huh",
        "okay",
        "right",
        "yeah",
        "yes",
        "exactly",
    )
    laughter_tokens: tuple[str, ...] = ("ha", "haha", "hehe", "laugh", "laughter")
    question_starters: tuple[str, ...] = ("how", "what", "why", "when", "where", "who")
    explicit_repair_starters: tuple[str, ...] = (
        "no",
        "wait",
        "actually",
        "stop",
        "repeat",
        "again",
    )
    explicit_repair_phrases: tuple[str, ...] = (
        "say that again",
        "say it again",
        "hold on",
        "that's not",
        "that is not",
    )
    acknowledgement_continuation_words: tuple[str, ...] = ("and", "but", "though", "however")
    classification_deadline_ms: int = 500
    interruption_probability_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.classification_deadline_ms <= 0:
            raise ValueError("Overlap classification deadline must be positive.")
        if not 0.0 <= self.interruption_probability_threshold <= 1.0:
            raise ValueError("Interruption probability threshold must be between zero and one.")
        if not self.acknowledgement_phrases:
            raise ValueError("At least one acknowledgement phrase is required.")


@dataclass(frozen=True)
class OverlapEvidence:
    elapsed_ms: int
    speech_active: bool
    transcript: str
    transcript_event_id: str | None
    interruption_probability: float | None
    interruption_evidence_event_id: str | None

    def __post_init__(self) -> None:
        if self.elapsed_ms < 0:
            raise ValueError("Overlap elapsed time cannot be negative.")
        if self.interruption_probability is not None and not (
            0.0 <= self.interruption_probability <= 1.0
        ):
            raise ValueError("Interruption probability must be between zero and one.")


@dataclass(frozen=True)
class ProvisionalOverlapDecision:
    kind: OverlapResolutionKind
    reason: OverlapResolutionReason
    confidence: float
    causal_source: CausalSource
    causal_event_ids: tuple[str, ...]
    fast_path: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Overlap decision confidence must be between zero and one.")


@dataclass(frozen=True)
class OverlapMetricsReport:
    overlap_count: int
    cooperative_overlap_count: int
    competitive_overlap_count: int
    user_onset_to_duck_p95_ms: float | None
    user_onset_to_pause_p95_ms: float | None
    explicit_stop_p95_ms: float | None
    user_onset_to_resume_p95_ms: float | None
    cooperative_overlap_p95_ms: float | None
    competitive_overlap_p95_ms: float | None
    paused_buffer_age_p95_ms: float | None
    generated_source_sample_count: int


class OverlapMetrics:
    def __init__(self) -> None:
        self.overlap_count = 0
        self.cooperative_overlap_count = 0
        self.competitive_overlap_count = 0
        self.user_onset_to_duck_ms: list[float] = []
        self.user_onset_to_pause_ms: list[float] = []
        self.explicit_stop_ms: list[float] = []
        self.user_onset_to_resume_ms: list[float] = []
        self.cooperative_overlap_ms: list[float] = []
        self.competitive_overlap_ms: list[float] = []
        self.paused_buffer_age_ms: list[float] = []
        self.generated_source_sample_count = 0

    def record_started(self) -> None:
        self.overlap_count += 1

    def record_resolution(
        self,
        kind: OverlapResolutionKind,
        onset_monotonic_time_ns: int,
        decision_monotonic_time_ns: int,
        generated_source_sample_count: int,
    ) -> None:
        duration_ms = _nanoseconds_to_milliseconds(
            decision_monotonic_time_ns - onset_monotonic_time_ns
        )
        self.generated_source_sample_count += generated_source_sample_count
        if kind is OverlapResolutionKind.NON_FLOOR_TAKING:
            self.cooperative_overlap_count += 1
            self.cooperative_overlap_ms.append(duration_ms)
        else:
            self.competitive_overlap_count += 1
            self.competitive_overlap_ms.append(duration_ms)

    def record_duck(
        self,
        onset_monotonic_time_ns: int,
        acknowledged_monotonic_time_ns: int,
    ) -> None:
        self.user_onset_to_duck_ms.append(
            _nanoseconds_to_milliseconds(acknowledged_monotonic_time_ns - onset_monotonic_time_ns)
        )

    def record_pause(
        self,
        onset_monotonic_time_ns: int,
        acknowledged_monotonic_time_ns: int,
    ) -> None:
        self.user_onset_to_pause_ms.append(
            _nanoseconds_to_milliseconds(acknowledged_monotonic_time_ns - onset_monotonic_time_ns)
        )

    def record_cancel(
        self,
        onset_monotonic_time_ns: int,
        acknowledged_monotonic_time_ns: int,
        fast_path: bool,
    ) -> None:
        if fast_path:
            self.explicit_stop_ms.append(
                _nanoseconds_to_milliseconds(
                    acknowledged_monotonic_time_ns - onset_monotonic_time_ns
                )
            )

    def record_resume(
        self,
        onset_monotonic_time_ns: int,
        acknowledged_monotonic_time_ns: int,
        paused_at_monotonic_time_ns: int | None,
    ) -> None:
        self.user_onset_to_resume_ms.append(
            _nanoseconds_to_milliseconds(acknowledged_monotonic_time_ns - onset_monotonic_time_ns)
        )
        if paused_at_monotonic_time_ns is not None:
            self.paused_buffer_age_ms.append(
                _nanoseconds_to_milliseconds(
                    acknowledged_monotonic_time_ns - paused_at_monotonic_time_ns
                )
            )

    def report(self) -> OverlapMetricsReport:
        return OverlapMetricsReport(
            overlap_count=self.overlap_count,
            cooperative_overlap_count=self.cooperative_overlap_count,
            competitive_overlap_count=self.competitive_overlap_count,
            user_onset_to_duck_p95_ms=_percentile(self.user_onset_to_duck_ms, 0.95),
            user_onset_to_pause_p95_ms=_percentile(self.user_onset_to_pause_ms, 0.95),
            explicit_stop_p95_ms=_percentile(self.explicit_stop_ms, 0.95),
            user_onset_to_resume_p95_ms=_percentile(self.user_onset_to_resume_ms, 0.95),
            cooperative_overlap_p95_ms=_percentile(self.cooperative_overlap_ms, 0.95),
            competitive_overlap_p95_ms=_percentile(self.competitive_overlap_ms, 0.95),
            paused_buffer_age_p95_ms=_percentile(self.paused_buffer_age_ms, 0.95),
            generated_source_sample_count=self.generated_source_sample_count,
        )


class ProvisionalVadTranscriptOverlapPolicy:
    """Replaceable VAD-plus-transcript baseline for assistant/user overlap."""

    def __init__(self, config: ProvisionalOverlapPolicyConfig) -> None:
        self.config = config
        self._acknowledgements = {
            _normalize_phrase(phrase) for phrase in config.acknowledgement_phrases
        }
        self._laughter_tokens = {_normalize_phrase(token) for token in config.laughter_tokens}

    def classify(self, evidence: OverlapEvidence) -> ProvisionalOverlapDecision:
        transcript = _normalize_phrase(evidence.transcript)
        words = tuple(transcript.split())
        causal_event_ids = tuple(
            event_id
            for event_id in (
                evidence.transcript_event_id,
                evidence.interruption_evidence_event_id,
            )
            if event_id is not None
        )
        if (
            evidence.interruption_probability is not None
            and evidence.interruption_probability >= self.config.interruption_probability_threshold
        ):
            return ProvisionalOverlapDecision(
                kind=OverlapResolutionKind.FLOOR_TAKING,
                reason=OverlapResolutionReason.PREDICTED_INTERRUPTION,
                confidence=evidence.interruption_probability,
                causal_source=CausalSource.TURN_ADAPTER,
                causal_event_ids=causal_event_ids,
                fast_path=True,
            )
        if self._is_explicit_repair(transcript, words):
            return ProvisionalOverlapDecision(
                kind=OverlapResolutionKind.RESPONSE_REQUIRED,
                reason=OverlapResolutionReason.EXPLICIT_STOP_OR_REPAIR,
                confidence=0.95,
                causal_source=CausalSource.LEXICAL_CLASSIFIER,
                causal_event_ids=causal_event_ids,
                fast_path=True,
            )
        if words and words[0] in self.config.question_starters:
            return ProvisionalOverlapDecision(
                kind=OverlapResolutionKind.RESPONSE_REQUIRED,
                reason=OverlapResolutionReason.QUESTION,
                confidence=0.9,
                causal_source=CausalSource.LEXICAL_CLASSIFIER,
                causal_event_ids=causal_event_ids,
                fast_path=True,
            )
        acknowledgement = self._matching_acknowledgement(words)
        if acknowledgement is not None:
            acknowledgement_word_count = len(acknowledgement.split())
            if len(words) > acknowledgement_word_count:
                reason = (
                    OverlapResolutionReason.ACKNOWLEDGEMENT_CONTINUATION
                    if words[acknowledgement_word_count]
                    in self.config.acknowledgement_continuation_words
                    else OverlapResolutionReason.MEANINGFUL_LEXICAL_MATERIAL
                )
                return ProvisionalOverlapDecision(
                    kind=OverlapResolutionKind.FLOOR_TAKING,
                    reason=reason,
                    confidence=0.9,
                    causal_source=CausalSource.LEXICAL_CLASSIFIER,
                    causal_event_ids=causal_event_ids,
                    fast_path=True,
                )
            if not evidence.speech_active:
                return ProvisionalOverlapDecision(
                    kind=OverlapResolutionKind.NON_FLOOR_TAKING,
                    reason=OverlapResolutionReason.CLOSED_ACKNOWLEDGEMENT,
                    confidence=0.85,
                    causal_source=CausalSource.LEXICAL_CLASSIFIER,
                    causal_event_ids=causal_event_ids,
                    fast_path=False,
                )
        elif transcript and transcript in self._laughter_tokens:
            if not evidence.speech_active:
                return ProvisionalOverlapDecision(
                    kind=OverlapResolutionKind.NON_FLOOR_TAKING,
                    reason=OverlapResolutionReason.SHORT_LAUGHTER,
                    confidence=0.75,
                    causal_source=CausalSource.LEXICAL_CLASSIFIER,
                    causal_event_ids=causal_event_ids,
                    fast_path=False,
                )
        elif transcript:
            return ProvisionalOverlapDecision(
                kind=OverlapResolutionKind.RESPONSE_REQUIRED,
                reason=OverlapResolutionReason.MEANINGFUL_LEXICAL_MATERIAL,
                confidence=0.8,
                causal_source=CausalSource.LEXICAL_CLASSIFIER,
                causal_event_ids=causal_event_ids,
                fast_path=False,
            )
        if evidence.elapsed_ms >= self.config.classification_deadline_ms:
            if evidence.speech_active:
                return ProvisionalOverlapDecision(
                    kind=OverlapResolutionKind.FLOOR_TAKING,
                    reason=OverlapResolutionReason.SPEECH_DURATION_DEADLINE,
                    confidence=0.9,
                    causal_source=CausalSource.SILERO_VAD,
                    causal_event_ids=causal_event_ids,
                    fast_path=False,
                )
            return ProvisionalOverlapDecision(
                kind=OverlapResolutionKind.NON_FLOOR_TAKING,
                reason=OverlapResolutionReason.ACOUSTIC_BURST_ENDED,
                confidence=0.6,
                causal_source=CausalSource.SILERO_VAD,
                causal_event_ids=causal_event_ids,
                fast_path=False,
            )
        if not evidence.speech_active:
            return ProvisionalOverlapDecision(
                kind=OverlapResolutionKind.NON_FLOOR_TAKING,
                reason=OverlapResolutionReason.ACOUSTIC_BURST_ENDED,
                confidence=0.7,
                causal_source=CausalSource.SILERO_VAD,
                causal_event_ids=causal_event_ids,
                fast_path=False,
            )
        return ProvisionalOverlapDecision(
            kind=OverlapResolutionKind.UNRESOLVED,
            reason=OverlapResolutionReason.AWAITING_MORE_EVIDENCE,
            confidence=0.5,
            causal_source=CausalSource.SILERO_VAD,
            causal_event_ids=causal_event_ids,
            fast_path=False,
        )

    def _matching_acknowledgement(self, words: tuple[str, ...]) -> str | None:
        transcript = " ".join(words)
        matches = tuple(
            acknowledgement
            for acknowledgement in self._acknowledgements
            if transcript == acknowledgement or transcript.startswith(f"{acknowledgement} ")
        )
        if not matches:
            return None
        return max(matches, key=len)

    def _is_explicit_repair(self, transcript: str, words: tuple[str, ...]) -> bool:
        if words and words[0] in self.config.explicit_repair_starters:
            return True
        return any(transcript.startswith(phrase) for phrase in self.config.explicit_repair_phrases)


def _normalize_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold()))


def _nanoseconds_to_milliseconds(duration_ns: int) -> float:
    if duration_ns < 0:
        raise AssertionError("Overlap metrics timestamps must increase monotonically.")
    return duration_ns / 1_000_000


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(math.ceil(percentile * len(ordered)), 1)
    return ordered[rank - 1]
