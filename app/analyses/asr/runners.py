from __future__ import annotations

import importlib.metadata
import importlib.util
import time
from collections.abc import Callable

from app.analyses.asr.models import AsrModelMode, AsrTranscriberContext
from app.asr_quality.schemas import TranscriptionResult, Word

JsonScalar = str | int | float | bool | None
JsonRecord = dict[str, JsonScalar]

PARAKEET_IDENTIFIER = "nvidia/parakeet-tdt-0.6b-v3"
WHISPERX_IDENTIFIER = "whisperx/large-v3"


def transcribe_with_model(
    model_mode: AsrModelMode,
    context: AsrTranscriberContext,
) -> TranscriptionResult:
    runner = model_runners()[model_mode]
    try:
        return runner(context)
    except Exception as error:
        return failed_transcription(
            context=context,
            model_name=model_mode.value,
            model_identifier=model_identifier_for_mode(model_mode),
            package_names=package_names_for_mode(model_mode),
            error=f"{type(error).__name__}: {error}",
        )


def model_runners() -> dict[AsrModelMode, Callable[[AsrTranscriberContext], TranscriptionResult]]:
    return {
        AsrModelMode.PARAKEET_TDT: transcribe_with_parakeet,
        AsrModelMode.WHISPERX: transcribe_with_whisperx,
    }


def model_identifier_for_mode(model_mode: AsrModelMode) -> str:
    match model_mode:
        case AsrModelMode.PARAKEET_TDT:
            return PARAKEET_IDENTIFIER
        case AsrModelMode.WHISPERX:
            return WHISPERX_IDENTIFIER


def package_names_for_mode(model_mode: AsrModelMode) -> tuple[str, ...]:
    match model_mode:
        case AsrModelMode.PARAKEET_TDT:
            return ("torch", "transformers", "librosa")
        case AsrModelMode.WHISPERX:
            return ("torch", "whisperx", "faster-whisper")


def transcribe_with_parakeet(context: AsrTranscriberContext) -> TranscriptionResult:
    missing_packages = missing_package_names(("librosa",))
    transformers_available = importlib.util.find_spec("transformers") is not None
    if missing_packages or not transformers_available:
        unavailable = tuple(
            [*missing_packages, *(("transformers",) if not transformers_available else ())]
        )
        return failed_transcription(
            context=context,
            model_name=AsrModelMode.PARAKEET_TDT.value,
            model_identifier=PARAKEET_IDENTIFIER,
            package_names=("torch", "transformers", "librosa"),
            error=f"Missing optional ASR dependencies: {', '.join(unavailable)}",
        )

    import librosa
    import torch
    from transformers import AutoModelForTDT, AutoProcessor

    processing_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(PARAKEET_IDENTIFIER)
    model = AutoModelForTDT.from_pretrained(PARAKEET_IDENTIFIER, dtype="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    sample_rate = int(processor.feature_extractor.sampling_rate)
    model_loading_time_seconds = time.perf_counter() - load_start
    inference_start = time.perf_counter()
    audio, _sample_rate = librosa.load(context.audio_path, sr=sample_rate, mono=True)
    inputs = processor([audio], sampling_rate=sample_rate)
    inputs.to(model.device, dtype=model.dtype)
    output = model.generate(**inputs, return_dict_in_generate=True)
    decoded_output, decoded_timestamps = processor.decode(
        output.sequences,
        durations=output.durations,
        skip_special_tokens=True,
    )
    inference_time_seconds = time.perf_counter() - inference_start
    processing_time_seconds = time.perf_counter() - processing_start
    raw_output = {
        "text": str(decoded_output),
        "timestamp_count": len(decoded_timestamps) if isinstance(decoded_timestamps, list) else 0,
    }
    return TranscriptionResult(
        model_name=AsrModelMode.PARAKEET_TDT.value,
        audio_path=context.audio_path,
        track=context.speaker_track,
        audio_duration_seconds=context.audio_duration_seconds,
        processing_time_seconds=processing_time_seconds,
        words=tuple(words_from_parakeet_timestamps(decoded_timestamps)),
        raw_output=raw_output,
        model_identifier=PARAKEET_IDENTIFIER,
        package_versions=package_versions(("torch", "transformers", "librosa")),
        model_loading_time_seconds=model_loading_time_seconds,
        inference_time_seconds=inference_time_seconds,
        real_time_factor=processing_time_seconds / context.audio_duration_seconds
        if context.audio_duration_seconds > 0.0
        else None,
        peak_gpu_memory_mb=peak_gpu_memory_mb(),
        error=None,
    )


def transcribe_with_whisperx(context: AsrTranscriberContext) -> TranscriptionResult:
    missing_packages = missing_package_names(("whisperx", "faster_whisper"))
    if missing_packages:
        return failed_transcription(
            context=context,
            model_name=AsrModelMode.WHISPERX.value,
            model_identifier=WHISPERX_IDENTIFIER,
            package_names=("torch", "whisperx", "faster-whisper"),
            error=f"Missing optional ASR dependencies: {', '.join(missing_packages)}",
        )

    import torch
    import whisperx
    from faster_whisper import WhisperModel

    processing_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    load_start = time.perf_counter()
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    model_loading_time_seconds = time.perf_counter() - load_start
    inference_start = time.perf_counter()
    segments, transcription_info = model.transcribe(
        context.audio_path,
        beam_size=5,
        word_timestamps=False,
        vad_filter=False,
    )
    transcription_segments = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in segments
    ]
    language_code = str(transcription_info.language)
    audio = whisperx.load_audio(context.audio_path)
    align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    aligned = whisperx.align(
        transcription_segments,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    inference_time_seconds = time.perf_counter() - inference_start
    processing_time_seconds = time.perf_counter() - processing_start
    return TranscriptionResult(
        model_name=AsrModelMode.WHISPERX.value,
        audio_path=context.audio_path,
        track=context.speaker_track,
        audio_duration_seconds=context.audio_duration_seconds,
        processing_time_seconds=processing_time_seconds,
        words=tuple(words_from_whisperx_output(aligned)),
        raw_output={
            "language": language_code,
            "segment_count": len(transcription_segments),
        },
        model_identifier=WHISPERX_IDENTIFIER,
        package_versions=package_versions(("torch", "whisperx", "faster-whisper")),
        model_loading_time_seconds=model_loading_time_seconds,
        inference_time_seconds=inference_time_seconds,
        real_time_factor=processing_time_seconds / context.audio_duration_seconds
        if context.audio_duration_seconds > 0.0
        else None,
        peak_gpu_memory_mb=peak_gpu_memory_mb(),
        error=None,
    )


def failed_transcription(
    context: AsrTranscriberContext,
    model_name: str,
    model_identifier: str,
    package_names: tuple[str, ...],
    error: str,
) -> TranscriptionResult:
    return TranscriptionResult(
        model_name=model_name,
        audio_path=context.audio_path,
        track=context.speaker_track,
        audio_duration_seconds=context.audio_duration_seconds,
        processing_time_seconds=0.0,
        words=(),
        raw_output={"error": error},
        model_identifier=model_identifier,
        package_versions=package_versions(package_names),
        error=error,
    )


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


def missing_package_names(package_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        package_name
        for package_name in package_names
        if importlib.util.find_spec(package_name) is None
    )


def package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def peak_gpu_memory_mb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1_048_576
