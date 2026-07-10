from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from app.analyses.end_of_turn.base import EndOfTurnDetectorInfo, EndOfTurnDetectorMode
from app.analyses.end_of_turn.service import BaselineResult, EndOfTurnEvent, SpeechSegment
from app.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS, read_mono_wave_audio

MODEL_REPOSITORY = "livekit/turn-detector"
MODEL_REVISION = "v0.4.1-intl"
MODEL_FILENAME = "model_q8.onnx"
LANGUAGES_FILENAME = "languages.json"
MAX_HISTORY_TOKENS = 128
MAX_HISTORY_TURNS = 6
DEFAULT_LANGUAGE_CODE = "en"
DEFAULT_ENGLISH_THRESHOLD = 0.011


class _FutureEndOfTurnDetectorMode(StrEnum):
    LIVEKIT_TEXT_TURN = "livekit_text_turn"


LIVEKIT_TEXT_TURN_MODE = cast(
    EndOfTurnDetectorMode,
    _FutureEndOfTurnDetectorMode.LIVEKIT_TEXT_TURN,
)


class ChatMessagePayload(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class TurnTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[ChatMessagePayload],
        add_generation_prompt: bool,
        add_special_tokens: bool,
        tokenize: Literal[False],
    ) -> str:
        raise NotImplementedError

    def __call__(
        self,
        text: str,
        add_special_tokens: bool,
        return_tensors: Literal["np"],
        max_length: int,
        truncation: bool,
    ) -> BatchEncoding:
        raise NotImplementedError


class TurnDetectorSession(Protocol):
    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.int64]],
    ) -> list[NDArray[np.float32]]:
        raise NotImplementedError


class TurnCompletionScorer(Protocol):
    def completion_probability(self, chat_messages: list[ChatMessagePayload]) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TranscriptTurn:
    speaker: str
    text: str
    start_seconds: float
    end_seconds: float
    words: tuple[TranscriptWord, ...]


@dataclass(frozen=True)
class CandidateTranscriptTurn:
    speech_start_seconds: float
    speech_end_seconds: float
    evaluate_seconds: float
    following_silence_seconds: float
    chat_messages: list[ChatMessagePayload]


@dataclass(frozen=True)
class LiveKitTextTurnInference:
    tokenizer: TurnTokenizer
    session: TurnDetectorSession

    def completion_probability(self, chat_messages: list[ChatMessagePayload]) -> float:
        text = _format_chat_messages(tokenizer=self.tokenizer, chat_messages=chat_messages)
        inputs = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        )
        input_ids = cast(NDArray[np.int64], inputs["input_ids"]).astype(np.int64)
        outputs = self.session.run(None, {"input_ids": input_ids})
        probabilities = outputs[0].flatten()
        return float(probabilities[-1].item())


@dataclass(frozen=True)
class LiveKitTextTurnDetector:
    info: EndOfTurnDetectorInfo
    candidate_silence_seconds: float
    threshold_override: float | None

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        return run_livekit_text_turn(
            speaker1_path=speaker1_path,
            result_name=self.info.mode.value,
            description=self.info.description,
            candidate_silence_seconds=self.candidate_silence_seconds,
            threshold_override=self.threshold_override,
        )


def livekit_text_turn_detector() -> LiveKitTextTurnDetector:
    return LiveKitTextTurnDetector(
        info=EndOfTurnDetectorInfo(
            mode=LIVEKIT_TEXT_TURN_MODE,
            label="LiveKit Text Turn",
            description=(
                "LUEL word-level transcript turns propose speaker-1 candidate boundaries after "
                "500 ms of speaker-1 silence; LiveKit's multilingual text turn detector "
                "scores the recent Qwen-formatted dialogue context and emits an end-of-turn "
                "event when the calibrated language threshold is met."
            ),
        ),
        candidate_silence_seconds=0.5,
        threshold_override=None,
    )


def run_livekit_text_turn(
    speaker1_path: Path,
    result_name: str,
    description: str,
    candidate_silence_seconds: float = 0.5,
    threshold_override: float | None = None,
) -> BaselineResult:
    transcript = _load_transcript_for_wave(speaker_path=speaker1_path)
    speaker_label = _speaker_label_from_wave_path(speaker_path=speaker1_path)
    language_code = _language_code(language_name=transcript.language_name)
    thresholds = _load_language_thresholds(
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
    )
    if threshold_override is not None:
        threshold = threshold_override
    else:
        threshold = thresholds.get(language_code, DEFAULT_ENGLISH_THRESHOLD)
    speech_segments = _speaker_speech_segments(
        transcript_turns=transcript.turns,
        speaker_label=speaker_label,
    )
    candidates = _candidate_transcript_turns(
        transcript_turns=transcript.turns,
        speaker_label=speaker_label,
        audio_duration_seconds=transcript.audio_duration_seconds,
        candidate_silence_seconds=candidate_silence_seconds,
    )
    inference = _load_livekit_text_turn_inference(
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
    )
    end_of_turn_events = _livekit_text_turn_events(
        candidate_turns=candidates,
        scorer=inference,
        completion_threshold=threshold,
    )
    return BaselineResult(
        name=result_name,
        description=description,
        frame_seconds=0.0,
        min_silence_seconds=candidate_silence_seconds,
        threshold=threshold,
        speech_segments=speech_segments,
        end_of_turn_events=end_of_turn_events,
    )


@dataclass(frozen=True)
class Transcript:
    language_name: str
    audio_duration_seconds: float
    turns: list[TranscriptTurn]


def _load_transcript_for_wave(speaker_path: Path) -> Transcript:
    metadata_path = _metadata_path_for_wave_path(speaker_path=speaker_path)
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = _required_mapping(value=raw_metadata, name=str(metadata_path))
    language_name = _optional_string(source=metadata, key="language", default="English")
    wave_audio = read_mono_wave_audio(wave_path=speaker_path)
    capped_duration_seconds = min(
        wave_audio.duration_seconds,
        ANALYSIS_AUDIO_MAX_DURATION_SECONDS,
    )
    turns = _transcript_turns(
        metadata=metadata,
        max_duration_seconds=capped_duration_seconds,
    )
    return Transcript(
        language_name=language_name,
        audio_duration_seconds=capped_duration_seconds,
        turns=turns,
    )


def _metadata_path_for_wave_path(speaker_path: Path) -> Path:
    match = re.fullmatch(r"(?P<session>.+)_speaker(?P<speaker_number>[12])", speaker_path.stem)
    if match is None:
        raise ValueError(
            f"Expected LUEL speaker WAV name like `session_speaker1.wav`: {speaker_path.name}"
        )
    session_identifier = match.group("session")
    metadata_path = speaker_path.with_name(f"{session_identifier}.json")
    if not metadata_path.exists():
        raise ValueError(f"Missing LUEL metadata JSON beside WAV: {metadata_path}")
    return metadata_path


def _speaker_label_from_wave_path(speaker_path: Path) -> str:
    match = re.fullmatch(r".+_speaker(?P<speaker_number>[12])", speaker_path.stem)
    if match is None:
        raise ValueError(
            f"Expected LUEL speaker WAV name like `session_speaker1.wav`: {speaker_path.name}"
        )
    return f"Speaker{match.group('speaker_number')}"


def _transcript_turns(
    metadata: Mapping[str, object],
    max_duration_seconds: float,
) -> list[TranscriptTurn]:
    raw_turns = _required_sequence(source=metadata, key="speakerTranscript")
    transcript_turns: list[TranscriptTurn] = []
    for raw_turn in raw_turns:
        turn = _required_mapping(value=raw_turn, name="speakerTranscript[]")
        start_seconds = _required_number(source=turn, key="startTime")
        if start_seconds >= max_duration_seconds:
            continue
        transcript_turn = _transcript_turn(
            turn=turn,
            max_duration_seconds=max_duration_seconds,
        )
        if transcript_turn.text:
            transcript_turns.append(transcript_turn)
    return transcript_turns


def _transcript_turn(
    turn: Mapping[str, object],
    max_duration_seconds: float,
) -> TranscriptTurn:
    words = _transcript_words(
        turn=turn,
        max_duration_seconds=max_duration_seconds,
    )
    speaker = _required_string(source=turn, key="speaker")
    fallback_text = _required_string(source=turn, key="text")
    start_seconds = _required_number(source=turn, key="startTime")
    end_seconds = min(_required_number(source=turn, key="endTime"), max_duration_seconds)
    if words:
        text = " ".join(word.text for word in words)
        return TranscriptTurn(
            speaker=speaker,
            text=text,
            start_seconds=words[0].start_seconds,
            end_seconds=words[-1].end_seconds,
            words=tuple(words),
        )
    return TranscriptTurn(
        speaker=speaker,
        text=fallback_text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        words=(),
    )


def _transcript_words(
    turn: Mapping[str, object],
    max_duration_seconds: float,
) -> list[TranscriptWord]:
    raw_words = _optional_sequence(source=turn, key="words")
    words: list[TranscriptWord] = []
    for raw_word in raw_words:
        word = _required_mapping(value=raw_word, name="words[]")
        start_seconds = _required_number(source=word, key="startTime")
        if start_seconds >= max_duration_seconds:
            continue
        end_seconds = min(_required_number(source=word, key="endTime"), max_duration_seconds)
        words.append(
            TranscriptWord(
                text=_required_string(source=word, key="text"),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return words


def _speaker_speech_segments(
    transcript_turns: list[TranscriptTurn],
    speaker_label: str,
) -> list[SpeechSegment]:
    return [
        SpeechSegment(
            start_seconds=turn.start_seconds,
            end_seconds=turn.end_seconds,
        )
        for turn in transcript_turns
        if turn.speaker == speaker_label
    ]


def _candidate_transcript_turns(
    transcript_turns: list[TranscriptTurn],
    speaker_label: str,
    audio_duration_seconds: float,
    candidate_silence_seconds: float,
) -> list[CandidateTranscriptTurn]:
    candidates: list[CandidateTranscriptTurn] = []
    for turn_index, transcript_turn in enumerate(transcript_turns):
        if transcript_turn.speaker != speaker_label:
            continue
        following_silence_seconds = _following_speaker_silence_seconds(
            transcript_turns=transcript_turns,
            speaker_label=speaker_label,
            turn_index=turn_index,
            audio_duration_seconds=audio_duration_seconds,
        )
        if following_silence_seconds < candidate_silence_seconds:
            continue
        candidates.append(
            CandidateTranscriptTurn(
                speech_start_seconds=transcript_turn.start_seconds,
                speech_end_seconds=transcript_turn.end_seconds,
                evaluate_seconds=_rounded_seconds(
                    transcript_turn.end_seconds + candidate_silence_seconds
                ),
                following_silence_seconds=_rounded_seconds(following_silence_seconds),
                chat_messages=_chat_messages_for_turn(
                    transcript_turns=transcript_turns,
                    speaker_label=speaker_label,
                    turn_index=turn_index,
                ),
            )
        )
    return candidates


def _following_speaker_silence_seconds(
    transcript_turns: list[TranscriptTurn],
    speaker_label: str,
    turn_index: int,
    audio_duration_seconds: float,
) -> float:
    current_turn = transcript_turns[turn_index]
    for following_turn in transcript_turns[turn_index + 1 :]:
        if following_turn.speaker == speaker_label:
            return following_turn.start_seconds - current_turn.end_seconds
    return audio_duration_seconds - current_turn.end_seconds


def _chat_messages_for_turn(
    transcript_turns: list[TranscriptTurn],
    speaker_label: str,
    turn_index: int,
) -> list[ChatMessagePayload]:
    messages: list[ChatMessagePayload] = []
    for transcript_turn in transcript_turns[: turn_index + 1]:
        role = "user" if transcript_turn.speaker == speaker_label else "assistant"
        messages.append(
            {
                "role": role,
                "content": transcript_turn.text,
            }
        )
    return messages[-MAX_HISTORY_TURNS:]


def _livekit_text_turn_events(
    candidate_turns: list[CandidateTranscriptTurn],
    scorer: TurnCompletionScorer,
    completion_threshold: float,
) -> list[EndOfTurnEvent]:
    events: list[EndOfTurnEvent] = []
    for candidate_turn in candidate_turns:
        completion_probability = scorer.completion_probability(candidate_turn.chat_messages)
        if completion_probability >= completion_threshold:
            events.append(
                EndOfTurnEvent(
                    time_seconds=candidate_turn.evaluate_seconds,
                    speech_start_seconds=candidate_turn.speech_start_seconds,
                    speech_end_seconds=candidate_turn.speech_end_seconds,
                    silence_seconds=candidate_turn.following_silence_seconds,
                )
            )
    return events


@lru_cache(maxsize=1)
def _load_livekit_text_turn_inference(
    model_repository: str,
    model_revision: str,
) -> LiveKitTextTurnInference:
    try:
        model_path = hf_hub_download(
            repo_id=model_repository,
            filename=MODEL_FILENAME,
            subfolder="onnx",
            revision=model_revision,
        )
        tokenizer = cast(
            TurnTokenizer,
            AutoTokenizer.from_pretrained(
                model_repository,
                revision=model_revision,
                truncation_side="left",
            ),
        )
    except OSError as error:
        raise ValueError(
            f"Could not load LiveKit text turn detector `{model_repository}` "
            f"at revision `{model_revision}`. Check network access, Hugging Face cache "
            "configuration, and model availability."
        ) from error

    session_options = ort.SessionOptions()
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = cast(
        TurnDetectorSession,
        ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        ),
    )
    return LiveKitTextTurnInference(tokenizer=tokenizer, session=session)


@lru_cache(maxsize=1)
def _load_language_thresholds(
    model_repository: str,
    model_revision: str,
) -> dict[str, float]:
    try:
        languages_path = hf_hub_download(
            repo_id=model_repository,
            filename=LANGUAGES_FILENAME,
            revision=model_revision,
        )
    except OSError as error:
        raise ValueError(
            f"Could not load LiveKit language thresholds from `{model_repository}` "
            f"at revision `{model_revision}`."
        ) from error

    raw_languages = json.loads(Path(languages_path).read_text(encoding="utf-8"))
    languages = _required_mapping(value=raw_languages, name=LANGUAGES_FILENAME)
    thresholds: dict[str, float] = {}
    for language_code, raw_language_config in languages.items():
        language_config = _required_mapping(
            value=raw_language_config,
            name=f"{LANGUAGES_FILENAME}:{language_code}",
        )
        thresholds[language_code] = _required_number(source=language_config, key="threshold")
    return thresholds


def _format_chat_messages(
    tokenizer: TurnTokenizer,
    chat_messages: list[ChatMessagePayload],
) -> str:
    normalized_messages = _normalized_chat_messages(chat_messages=chat_messages)
    conversation_text = tokenizer.apply_chat_template(
        normalized_messages,
        add_generation_prompt=False,
        add_special_tokens=False,
        tokenize=False,
    )
    end_token_index = conversation_text.rfind("<|im_end|>")
    if end_token_index < 0:
        raise ValueError("LiveKit text turn tokenizer did not emit an end-of-message token.")
    return conversation_text[:end_token_index]


def _normalized_chat_messages(
    chat_messages: list[ChatMessagePayload],
) -> list[ChatMessagePayload]:
    normalized_messages: list[ChatMessagePayload] = []
    for chat_message in chat_messages:
        normalized_content = _normalize_text(text=chat_message["content"])
        if not normalized_content:
            continue
        if normalized_messages and normalized_messages[-1]["role"] == chat_message["role"]:
            normalized_messages[-1]["content"] += f" {normalized_content}"
            continue
        normalized_messages.append(
            {
                "role": chat_message["role"],
                "content": normalized_content,
            }
        )
    return normalized_messages


def _normalize_text(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKC", text.lower())
    unpunctuated_text = "".join(
        character
        for character in normalized_text
        if not (unicodedata.category(character).startswith("P") and character not in {"'", "-"})
    )
    return re.sub(r"\s+", " ", unpunctuated_text).strip()


def _language_code(language_name: str) -> str:
    normalized_language_name = language_name.strip().lower()
    match normalized_language_name:
        case "english" | "en" | "en-us" | "en_us":
            return "en"
        case "spanish" | "es":
            return "es"
        case "french" | "fr":
            return "fr"
        case "german" | "de":
            return "de"
        case "italian" | "it":
            return "it"
        case "portuguese" | "pt":
            return "pt"
        case "dutch" | "nl":
            return "nl"
        case "chinese" | "zh":
            return "zh"
        case "japanese" | "ja":
            return "ja"
        case "korean" | "ko":
            return "ko"
        case "indonesian" | "id":
            return "id"
        case "turkish" | "tr":
            return "tr"
        case "russian" | "ru":
            return "ru"
        case "hindi" | "hi":
            return "hi"
        case _:
            return DEFAULT_LANGUAGE_CODE


def _required_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected `{name}` to be an object.")
    return cast(Mapping[str, object], value)


def _required_sequence(source: Mapping[str, object], key: str) -> Sequence[object]:
    try:
        value = source[key]
    except KeyError as error:
        raise ValueError(f"Missing required `{key}` field.") from error
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Expected `{key}` to be an array.")
    return cast(Sequence[object], value)


def _optional_sequence(source: Mapping[str, object], key: str) -> Sequence[object]:
    value = source.get(key)
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Expected `{key}` to be an array.")
    return cast(Sequence[object], value)


def _required_string(source: Mapping[str, object], key: str) -> str:
    try:
        value = source[key]
    except KeyError as error:
        raise ValueError(f"Missing required `{key}` field.") from error
    if not isinstance(value, str):
        raise ValueError(f"Expected `{key}` to be a string.")
    return value


def _optional_string(
    source: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    value = source.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"Expected `{key}` to be a string.")
    return value


def _required_number(source: Mapping[str, object], key: str) -> float:
    try:
        value = source[key]
    except KeyError as error:
        raise ValueError(f"Missing required `{key}` field.") from error
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Expected `{key}` to be a number.")
    return float(value)


def _rounded_seconds(seconds: float) -> float:
    return round(seconds, 6)
