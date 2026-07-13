from __future__ import annotations

from typing import TypeAlias

from app.asr.transcript import Word

JsonScalar: TypeAlias = str | int | float | bool | None
JsonRecord: TypeAlias = dict[str, JsonScalar]

PARAKEET_IDENTIFIER = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
WHISPER_IDENTIFIER = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
CANARY_IDENTIFIER = "nvidia/canary-1b-v2"
CANARY_REVISION = "87bc52657add533cd0156b3fc1aef027280754bf"
NEMOTRON_3_5_IDENTIFIER = "nvidia/nemotron-3.5-asr-streaming-0.6b"
NEMOTRON_3_5_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"


def words_from_parakeet_timestamps(decoded_timestamps: object) -> list[Word]:
    if (
        isinstance(decoded_timestamps, list)
        and decoded_timestamps
        and isinstance(decoded_timestamps[0], list)
    ):
        decoded_timestamps = decoded_timestamps[0]
    if not isinstance(decoded_timestamps, list):
        raise ValueError("Parakeet timestamps must be a list")
    return merge_timestamp_pieces(timestamp_pieces(decoded_timestamps))


def timestamp_pieces(raw_items: list[object]) -> list[Word]:
    words: list[Word] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Parakeet timestamp item must be an object")
        payload = scalar_record(item)
        words.append(
            Word(
                text=string_from_keys(payload, ("word", "text", "token")),
                start_seconds=float_from_keys(payload, ("start", "start_time", "start_offset")),
                end_seconds=float_from_keys(payload, ("end", "end_time", "end_offset")),
                confidence=float_from_keys(payload, ("confidence", "score")),
            )
        )
    return words


def merge_timestamp_pieces(pieces: list[Word]) -> list[Word]:
    words: list[Word] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None
    for piece in pieces:
        starts_new_word = bool(piece.text[:1].isspace()) or current_text == ""
        cleaned_text = piece.text.strip()
        if not cleaned_text:
            continue
        if starts_new_word and current_text:
            words.append(
                Word(text=current_text, start_seconds=current_start, end_seconds=current_end)
            )
            current_text = ""
            current_start = None
            current_end = None
        current_text += cleaned_text
        current_start = current_start if current_start is not None else piece.start_seconds
        if piece.end_seconds is not None:
            current_end = (
                max(current_end, piece.end_seconds)
                if current_end is not None
                else piece.end_seconds
            )
    if current_text:
        words.append(Word(text=current_text, start_seconds=current_start, end_seconds=current_end))
    return words


def words_from_whisperx_output(aligned_output: object) -> list[Word]:
    if not isinstance(aligned_output, dict):
        raise ValueError("WhisperX aligned output must be an object")
    segments = aligned_output.get("segments")
    if not isinstance(segments, list):
        raise ValueError("WhisperX aligned output is missing segments")
    words: list[Word] = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("WhisperX segment must be an object")
        segment_words = segment.get("words")
        if not isinstance(segment_words, list):
            continue
        for word_payload in segment_words:
            if not isinstance(word_payload, dict):
                raise ValueError("WhisperX word must be an object")
            payload = scalar_record(word_payload)
            words.append(
                Word(
                    text=string_from_keys(payload, ("word",)).strip(),
                    start_seconds=float_from_keys(payload, ("start",)),
                    end_seconds=float_from_keys(payload, ("end",)),
                    confidence=float_from_keys(payload, ("score",)),
                )
            )
    return words


def words_from_nemo_timestamps(timestamp_payloads: list[dict[str, JsonScalar]]) -> list[Word]:
    words: list[Word] = []
    for timestamp_payload in timestamp_payloads:
        words.append(
            Word(
                text=string_from_keys(timestamp_payload, ("word", "text")),
                start_seconds=float_from_keys(timestamp_payload, ("start", "start_offset")),
                end_seconds=float_from_keys(timestamp_payload, ("end", "end_offset")),
                confidence=float_from_keys(timestamp_payload, ("confidence", "score")),
            )
        )
    return words


def scalar_record(payload: dict[object, object]) -> JsonRecord:
    record: JsonRecord = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str | int | float | bool) or value is None:
            record[key] = value
    return record


def string_from_keys(payload: JsonRecord, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(f"Missing string field from keys {keys}")


def float_from_keys(payload: JsonRecord, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float):
            return float(value)
        if value is None and key in payload:
            return None
    return None
