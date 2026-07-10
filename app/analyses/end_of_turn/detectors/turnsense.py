from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

TURNSENSE_MODEL_ID = "latishab/turnsense"
TURNSENSE_MODEL_FILENAME = "model_quantized.onnx"
TURNSENSE_MAX_LENGTH = 256
TURNSENSE_MIN_SILENCE_SECONDS = 0.5
TURNSENSE_EOU_THRESHOLD = 0.5
ROUND_SECONDS_DIGITS = 6


class TurnsenseLabel(StrEnum):
    NON_EOU = "NON_EOU"
    EOU = "EOU"


@dataclass(frozen=True)
class TurnsensePrediction:
    label: TurnsenseLabel
    eou_probability: float


class TurnsenseTextClassifier(Protocol):
    def predict_end_of_turn(self, text: str) -> TurnsensePrediction:
        raise NotImplementedError


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class SpeakerTranscript:
    speaker: str
    words: list[TranscriptWord]


@dataclass(frozen=True)
class SessionTranscript:
    name: str
    duration_seconds: float
    speaker_transcripts: list[SpeakerTranscript]


@dataclass(frozen=True)
class SpeakerMetadataLocation:
    metadata_path: Path
    speaker_name: str


@dataclass(frozen=True)
class CandidateBoundary:
    word_index: int
    silence_seconds: float


class TurnsenseOnnxTextClassifier:
    def __init__(
        self,
        model_id: str = TURNSENSE_MODEL_ID,
        model_filename: str = TURNSENSE_MODEL_FILENAME,
        max_length: int = TURNSENSE_MAX_LENGTH,
    ) -> None:
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)
        model_path = hf_hub_download(repo_id=model_id, filename=model_filename)
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._max_length = max_length
        self._validate_session_inputs()

    def predict_end_of_turn(self, text: str) -> TurnsensePrediction:
        tokenized_inputs = self._tokenizer(
            f"<|user|> {text}",
            padding="max_length",
            max_length=self._max_length,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
        }
        logits = self._session.run(None, ort_inputs)[0]
        if logits.shape != (1, 2):
            raise ValueError(f"Unexpected Turnsense logits shape: {logits.shape}.")

        probabilities = _softmax(logits=logits[0])
        eou_probability = float(probabilities[1])
        if eou_probability >= TURNSENSE_EOU_THRESHOLD:
            return TurnsensePrediction(
                label=TurnsenseLabel.EOU,
                eou_probability=eou_probability,
            )
        return TurnsensePrediction(
            label=TurnsenseLabel.NON_EOU,
            eou_probability=eou_probability,
        )

    def _validate_session_inputs(self) -> None:
        input_names = {model_input.name for model_input in self._session.get_inputs()}
        expected_input_names = {"input_ids", "attention_mask"}
        if input_names != expected_input_names:
            raise ValueError(
                "Unexpected Turnsense ONNX inputs. "
                f"Expected {sorted(expected_input_names)}, got {sorted(input_names)}."
            )


@dataclass(frozen=True)
class TurnsenseDetector:
    info: EndOfTurnDetectorInfo
    classifier: TurnsenseTextClassifier
    min_silence_seconds: float
    eou_threshold: float

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        metadata_location = _speaker_metadata_location(wave_path=speaker1_path)
        session_transcript = _read_session_transcript(metadata_path=metadata_location.metadata_path)
        analysis_end_seconds = min(
            session_transcript.duration_seconds,
            ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
        )
        words = _speaker_words(
            session_transcript=session_transcript,
            speaker_name=metadata_location.speaker_name,
            analysis_end_seconds=analysis_end_seconds,
        )
        classified_turns = _classified_turns(
            words=words,
            analysis_end_seconds=analysis_end_seconds,
            classifier=self.classifier,
            min_silence_seconds=self.min_silence_seconds,
            eou_threshold=self.eou_threshold,
        )

        return BaselineResult(
            name=self.info.mode.value,
            description=self.info.description,
            frame_seconds=0.0,
            min_silence_seconds=self.min_silence_seconds,
            threshold=self.eou_threshold,
            speech_segments=classified_turns.speech_segments,
            end_of_turn_events=classified_turns.end_of_turn_events,
        )


@dataclass(frozen=True)
class ClassifiedTurns:
    speech_segments: list[SpeechSegment]
    end_of_turn_events: list[EndOfTurnEvent]


def turnsense_detector() -> TurnsenseDetector:
    try:
        mode = EndOfTurnDetectorMode("turnsense")
    except ValueError as error:
        raise ValueError(
            "EndOfTurnDetectorMode.TURNSENSE must be added with value "
            "`turnsense` before wiring turnsense_detector()."
        ) from error

    return TurnsenseDetector(
        info=EndOfTurnDetectorInfo(
            mode=mode,
            label="Turnsense",
            description=(
                "Transcript-only speaker 1 end-of-utterance detection using "
                "latishab/turnsense SmolLM2-135M quantized ONNX over LUEL word-timing "
                "candidate boundaries inside the 180 s analysis cap."
            ),
        ),
        classifier=TurnsenseOnnxTextClassifier(),
        min_silence_seconds=TURNSENSE_MIN_SILENCE_SECONDS,
        eou_threshold=TURNSENSE_EOU_THRESHOLD,
    )


def _classified_turns(
    words: list[TranscriptWord],
    analysis_end_seconds: float,
    classifier: TurnsenseTextClassifier,
    min_silence_seconds: float,
    eou_threshold: float,
) -> ClassifiedTurns:
    speech_segments: list[SpeechSegment] = []
    end_of_turn_events: list[EndOfTurnEvent] = []
    turn_start_index = 0

    for boundary in _candidate_boundaries(
        words=words,
        analysis_end_seconds=analysis_end_seconds,
        min_silence_seconds=min_silence_seconds,
    ):
        text = _turn_text(words=words[turn_start_index : boundary.word_index + 1])
        prediction = classifier.predict_end_of_turn(text=text)
        if prediction.label is not TurnsenseLabel.EOU or prediction.eou_probability < eou_threshold:
            continue

        speech_segment = SpeechSegment(
            start_seconds=_rounded_seconds(words[turn_start_index].start_seconds),
            end_seconds=_rounded_seconds(words[boundary.word_index].end_seconds),
        )
        speech_segments.append(speech_segment)
        end_of_turn_events.append(
            EndOfTurnEvent(
                time_seconds=_rounded_seconds(speech_segment.end_seconds + min_silence_seconds),
                speech_start_seconds=speech_segment.start_seconds,
                speech_end_seconds=speech_segment.end_seconds,
                silence_seconds=_rounded_seconds(boundary.silence_seconds),
            )
        )
        turn_start_index = boundary.word_index + 1

    if turn_start_index < len(words):
        speech_segments.append(
            SpeechSegment(
                start_seconds=_rounded_seconds(words[turn_start_index].start_seconds),
                end_seconds=_rounded_seconds(words[-1].end_seconds),
            )
        )

    return ClassifiedTurns(
        speech_segments=speech_segments,
        end_of_turn_events=end_of_turn_events,
    )


def _candidate_boundaries(
    words: list[TranscriptWord],
    analysis_end_seconds: float,
    min_silence_seconds: float,
) -> list[CandidateBoundary]:
    boundaries: list[CandidateBoundary] = []
    for word_index, word in enumerate(words):
        if word_index + 1 < len(words):
            next_start_seconds = words[word_index + 1].start_seconds
        else:
            next_start_seconds = analysis_end_seconds

        silence_seconds = next_start_seconds - word.end_seconds
        if silence_seconds >= min_silence_seconds:
            boundaries.append(
                CandidateBoundary(
                    word_index=word_index,
                    silence_seconds=silence_seconds,
                )
            )
    return boundaries


def _turn_text(words: list[TranscriptWord]) -> str:
    return " ".join(word.text for word in words)


def _speaker_metadata_location(wave_path: Path) -> SpeakerMetadataLocation:
    suffix = "_speaker1"
    if not wave_path.name.endswith(f"{suffix}.wav"):
        raise ValueError(f"Expected a speaker 1 WAV path ending in `{suffix}.wav`: {wave_path}")

    session_identifier = wave_path.stem[: -len(suffix)]
    if not session_identifier:
        raise ValueError(f"Could not derive session identifier from WAV path: {wave_path}")

    return SpeakerMetadataLocation(
        metadata_path=wave_path.parent / f"{session_identifier}.json",
        speaker_name="Speaker1",
    )


def _read_session_transcript(metadata_path: Path) -> SessionTranscript:
    if not metadata_path.exists():
        raise ValueError(f"Missing transcript metadata beside WAV: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata_payload = cast(JsonValue, json.load(metadata_file))

    return _parse_session_transcript(metadata_payload=metadata_payload, metadata_path=metadata_path)


def _parse_session_transcript(
    metadata_payload: JsonValue,
    metadata_path: Path,
) -> SessionTranscript:
    metadata_object = _required_object(
        value=metadata_payload,
        field_name=f"metadata root in {metadata_path}",
    )
    speaker_transcripts_value = _required_field(
        source=metadata_object,
        field_name="speakerTranscript",
    )
    speaker_transcripts_array = _required_array(
        value=speaker_transcripts_value,
        field_name="speakerTranscript",
    )
    if not speaker_transcripts_array:
        raise ValueError(f"Missing speakerTranscript entries in metadata: {metadata_path}")

    return SessionTranscript(
        name=_required_string(
            value=_required_field(source=metadata_object, field_name="name"),
            field_name="name",
        ),
        duration_seconds=_required_float(
            value=_required_field(source=metadata_object, field_name="durationSeconds"),
            field_name="durationSeconds",
        ),
        speaker_transcripts=[
            _parse_speaker_transcript(
                speaker_transcript_value=speaker_transcript_value,
                speaker_transcript_index=speaker_transcript_index,
            )
            for speaker_transcript_index, speaker_transcript_value in enumerate(
                speaker_transcripts_array
            )
        ],
    )


def _parse_speaker_transcript(
    speaker_transcript_value: JsonValue,
    speaker_transcript_index: int,
) -> SpeakerTranscript:
    field_prefix = f"speakerTranscript[{speaker_transcript_index}]"
    speaker_transcript_object = _required_object(
        value=speaker_transcript_value,
        field_name=field_prefix,
    )
    words_value = _required_field(
        source=speaker_transcript_object,
        field_name="words",
    )
    words_array = _required_array(
        value=words_value,
        field_name=f"{field_prefix}.words",
    )
    return SpeakerTranscript(
        speaker=_required_string(
            value=_required_field(source=speaker_transcript_object, field_name="speaker"),
            field_name=f"{field_prefix}.speaker",
        ),
        words=[
            _parse_transcript_word(
                word_value=word_value,
                word_index=word_index,
                field_prefix=field_prefix,
            )
            for word_index, word_value in enumerate(words_array)
        ],
    )


def _parse_transcript_word(
    word_value: JsonValue,
    word_index: int,
    field_prefix: str,
) -> TranscriptWord:
    word_field_prefix = f"{field_prefix}.words[{word_index}]"
    word_object = _required_object(value=word_value, field_name=word_field_prefix)
    start_seconds = _required_float(
        value=_required_field(source=word_object, field_name="startTime"),
        field_name=f"{word_field_prefix}.startTime",
    )
    end_seconds = _required_float(
        value=_required_field(source=word_object, field_name="endTime"),
        field_name=f"{word_field_prefix}.endTime",
    )
    if end_seconds < start_seconds:
        raise ValueError(f"Transcript word endTime precedes startTime at {word_field_prefix}.")

    return TranscriptWord(
        text=_required_string(
            value=_required_field(source=word_object, field_name="text"),
            field_name=f"{word_field_prefix}.text",
        ),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _speaker_words(
    session_transcript: SessionTranscript,
    speaker_name: str,
    analysis_end_seconds: float,
) -> list[TranscriptWord]:
    selected_transcripts = [
        speaker_transcript
        for speaker_transcript in session_transcript.speaker_transcripts
        if speaker_transcript.speaker == speaker_name
    ]
    if not selected_transcripts:
        raise ValueError(
            f"Missing speaker transcript for {speaker_name} in {session_transcript.name}."
        )

    words = [
        _clipped_word(word=word, analysis_end_seconds=analysis_end_seconds)
        for speaker_transcript in selected_transcripts
        for word in speaker_transcript.words
        if _word_overlaps_analysis_window(
            word=word,
            analysis_end_seconds=analysis_end_seconds,
        )
    ]
    clipped_words = [word for word in words if word is not None]
    if not clipped_words:
        raise ValueError(
            f"Speaker transcript for {speaker_name} in {session_transcript.name} "
            "has no timed words inside the analysis window."
        )
    return sorted(clipped_words, key=lambda word: (word.start_seconds, word.end_seconds))


def _word_overlaps_analysis_window(
    word: TranscriptWord,
    analysis_end_seconds: float,
) -> bool:
    return word.start_seconds < analysis_end_seconds and word.end_seconds > 0.0


def _clipped_word(
    word: TranscriptWord,
    analysis_end_seconds: float,
) -> TranscriptWord | None:
    start_seconds = max(0.0, word.start_seconds)
    end_seconds = min(analysis_end_seconds, word.end_seconds)
    if end_seconds <= start_seconds:
        return None
    return TranscriptWord(
        text=word.text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _required_field(source: dict[str, JsonValue], field_name: str) -> JsonValue:
    if field_name not in source:
        raise ValueError(f"Missing required metadata field `{field_name}`.")
    return source[field_name]


def _required_object(value: JsonValue, field_name: str) -> dict[str, JsonValue]:
    match value:
        case dict():
            return value
        case _:
            raise ValueError(f"Expected `{field_name}` to be an object.")


def _required_array(value: JsonValue, field_name: str) -> list[JsonValue]:
    match value:
        case list():
            return value
        case _:
            raise ValueError(f"Expected `{field_name}` to be an array.")


def _required_string(value: JsonValue, field_name: str) -> str:
    match value:
        case str():
            return value
        case _:
            raise ValueError(f"Expected `{field_name}` to be a string.")


def _required_float(value: JsonValue, field_name: str) -> float:
    match value:
        case int() | float():
            return float(value)
        case _:
            raise ValueError(f"Expected `{field_name}` to be a number.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits)


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, ROUND_SECONDS_DIGITS)
