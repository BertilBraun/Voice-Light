from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from src.models import (
    BackchannelPlacementDefinition,
    BackchannelPlacementOutput,
    InternalPausePlacementDefinition,
    InternalPausePlacementOutput,
    InterruptionPlacementDefinition,
    InterruptionPlacementOutput,
)


@dataclass(frozen=True)
class WordTimestamp:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class AnchorMatch:
    matched_text: str
    score: float
    start_index: int
    end_index: int
    start_seconds: float
    end_seconds: float
    last_word_start_seconds: float


class WhisperTimestampTranscriber:
    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: object | None = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise ValueError(
                "Automatic backchannel placement requires faster-whisper. Install with "
                "`uv sync --extra asr` or `uv pip install faster-whisper`."
            ) from error
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )

    def transcribe_words(self, audio_path: Path) -> list[WordTimestamp]:
        if self._model is None:
            self.load()
        assert self._model is not None
        segments, _transcription_info = self._model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=False,
        )
        words: list[WordTimestamp] = []
        for segment in segments:
            if segment.words is None:
                continue
            for word in segment.words:
                cleaned_word = word.word.strip()
                if cleaned_word == "":
                    continue
                words.append(
                    WordTimestamp(
                        text=cleaned_word,
                        start_seconds=float(word.start),
                        end_seconds=float(word.end),
                    )
                )
        if len(words) == 0:
            raise ValueError(f"ASR produced no word timestamps for {audio_path}")
        return words


def place_backchannel_from_words(
    words: list[WordTimestamp],
    placement_definition: BackchannelPlacementDefinition,
) -> tuple[float, BackchannelPlacementOutput]:
    anchor_words = normalize_words(placement_definition.anchor_text)
    if len(anchor_words) == 0:
        raise ValueError("Placement anchor_text must contain at least one searchable word.")
    match = find_anchor_match(words, anchor_words)
    suggested_offset_seconds = match.end_seconds + (placement_definition.delay_ms / 1000.0)
    output = BackchannelPlacementOutput(
        anchor_text=placement_definition.anchor_text,
        transcript_text=" ".join(word.text for word in words),
        matched_text=match.matched_text,
        score=match.score,
        anchor_start_seconds=match.start_seconds,
        anchor_end_seconds=match.end_seconds,
        delay_ms=placement_definition.delay_ms,
        speaker_b_start_seconds=suggested_offset_seconds,
    )
    return suggested_offset_seconds, output


def place_interruption_from_words(
    words: list[WordTimestamp],
    placement_definition: InterruptionPlacementDefinition,
) -> tuple[float, InterruptionPlacementOutput]:
    anchor_words = normalize_words(placement_definition.anchor_text)
    if len(anchor_words) == 0:
        raise ValueError("Placement anchor_text must contain at least one searchable word.")
    match = find_anchor_match(words, anchor_words)
    match placement_definition.mode:
        case "before_last_word":
            target_seconds = match.last_word_start_seconds - (placement_definition.lead_ms / 1000.0)
        case "at_anchor_start":
            target_seconds = match.start_seconds - (placement_definition.lead_ms / 1000.0)
        case "at_anchor_end":
            target_seconds = match.end_seconds - (placement_definition.lead_ms / 1000.0)
    target_seconds = max(0.0, target_seconds)
    output = InterruptionPlacementOutput(
        anchor_text=placement_definition.anchor_text,
        transcript_text=" ".join(word.text for word in words),
        matched_text=match.matched_text,
        score=match.score,
        anchor_start_seconds=match.start_seconds,
        anchor_end_seconds=match.end_seconds,
        speaker_b_start_seconds=target_seconds,
        mode=placement_definition.mode,
        lead_ms=placement_definition.lead_ms,
    )
    return target_seconds, output


def measure_internal_pause_from_words(
    words: list[WordTimestamp],
    placement_definition: InternalPausePlacementDefinition,
) -> InternalPausePlacementOutput:
    anchor_words = normalize_words(placement_definition.anchor_text)
    if len(anchor_words) == 0:
        raise ValueError("Placement anchor_text must contain at least one searchable word.")
    match = find_anchor_match(words, anchor_words)
    next_word_index = match.end_index
    if next_word_index >= len(words):
        raise ValueError(
            "Internal pause anchor "
            f"'{placement_definition.anchor_text}' matched the end of the transcript."
        )
    next_word = words[next_word_index]
    measured_pause_ms = int(round(max(0.0, next_word.start_seconds - match.end_seconds) * 1000))
    return InternalPausePlacementOutput(
        anchor_text=placement_definition.anchor_text,
        transcript_text=" ".join(word.text for word in words),
        matched_text=match.matched_text,
        score=match.score,
        anchor_start_seconds=match.start_seconds,
        anchor_end_seconds=match.end_seconds,
        next_word_text=next_word.text,
        next_word_start_seconds=next_word.start_seconds,
        measured_pause_ms=measured_pause_ms,
        pause=placement_definition.pause,
    )


def find_anchor_match(words: list[WordTimestamp], anchor_words: list[str]) -> AnchorMatch:
    normalized_transcript = [normalize_word(word.text) for word in words]
    target_length = len(anchor_words)
    minimum_window = max(1, target_length - 2)
    maximum_window = min(len(normalized_transcript), target_length + 3)
    best_match: AnchorMatch | None = None
    for window_length in range(minimum_window, maximum_window + 1):
        for start_index in range(0, len(normalized_transcript) - window_length + 1):
            end_index = start_index + window_length
            candidate_words = normalized_transcript[start_index:end_index]
            candidate_text = " ".join(candidate_words)
            score = SequenceMatcher(None, " ".join(anchor_words), candidate_text).ratio()
            if best_match is None or score > best_match.score:
                best_match = AnchorMatch(
                    matched_text=" ".join(word.text for word in words[start_index:end_index]),
                    score=score,
                    start_index=start_index,
                    end_index=end_index,
                    start_seconds=words[start_index].start_seconds,
                    end_seconds=words[end_index - 1].end_seconds,
                    last_word_start_seconds=words[end_index - 1].start_seconds,
                )
    assert best_match is not None
    if best_match.score < 0.55:
        raise ValueError(
            "Could not confidently match backchannel anchor "
            f"'{' '.join(anchor_words)}'. Best match '{best_match.matched_text}' "
            f"scored {best_match.score:.2f}."
        )
    return best_match


def normalize_words(text: str) -> list[str]:
    return [word for word in (normalize_word(part) for part in text.split()) if word != ""]


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
