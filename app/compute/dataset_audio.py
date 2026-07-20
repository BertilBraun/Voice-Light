from __future__ import annotations

import tempfile
from pathlib import Path

from app.compute.asr.models.base import prepare_asr_audio
from app.compute.asr.service import RemoteAsrModelCache, transcribe_requested_models
from app.shared.asr import AsrModelId
from app.shared.audio.s3 import CachedAudioFile, S3AudioCache
from app.shared.audio.transport import CANONICAL_MODEL_AUDIO_SPEC, prepare_audio_transport
from app.shared.compute_api import (
    DatasetAsrRequest,
    DatasetAsrResponse,
    DatasetLanguageRequest,
    DatasetLanguageResponse,
    DatasetLanguageTrackResponse,
    MaterializedAudio,
)
from app.shared.language import (
    active_probe_windows,
    prepare_language_probe_audio,
    track_language_status,
)


def analyze_dataset_language(
    request: DatasetLanguageRequest,
    audio_cache: S3AudioCache,
    model_cache: RemoteAsrModelCache,
) -> DatasetLanguageResponse:
    return DatasetLanguageResponse(
        speaker1=analyze_track_language(
            audio=audio_cache.materialize(request.speaker1),
            model_cache=model_cache,
        ),
        speaker2=analyze_track_language(
            audio=audio_cache.materialize(request.speaker2),
            model_cache=model_cache,
        ),
    )


def analyze_track_language(
    audio: CachedAudioFile,
    model_cache: RemoteAsrModelCache,
) -> DatasetLanguageTrackResponse:
    windows = active_probe_windows(audio.path)
    if not windows:
        return DatasetLanguageTrackResponse(
            audio=materialized_audio(audio),
            status=track_language_status(None, None, 0),
            language_code=None,
            confidence=None,
            transcript_word_count=0,
            transcript_text="",
            probe_windows=(),
            error=None,
        )
    output_file = tempfile.NamedTemporaryFile(
        prefix=f"{audio.path.stem}_language_probe_",
        suffix=".ogg",
        delete=False,
    )
    output_path = Path(output_file.name)
    output_file.close()
    try:
        prepare_language_probe_audio(
            source_path=audio.path,
            windows=windows,
            output_path=output_path,
        )
        results = transcribe_requested_models(
            model_cache=model_cache,
            model_ids=(AsrModelId.WHISPERX,),
            audio=prepare_asr_audio(output_path),
        )
    finally:
        output_path.unlink(missing_ok=True)
    if len(results) != 1:
        raise ValueError("Language assessment must return exactly one Whisper result.")
    transcript = results[0]
    estimate = transcript.language_estimate
    language_code = estimate.language_code if estimate is not None else None
    confidence = estimate.confidence if estimate is not None else None
    return DatasetLanguageTrackResponse(
        audio=materialized_audio(audio),
        status=track_language_status(
            language_code=language_code,
            confidence=confidence,
            transcript_word_count=len(transcript.words),
        ),
        language_code=language_code,
        confidence=confidence,
        transcript_word_count=len(transcript.words),
        transcript_text=transcript.text,
        probe_windows=windows,
        error=transcript.error,
    )


def transcribe_dataset_audio(
    request: DatasetAsrRequest,
    audio_cache: S3AudioCache,
    model_cache: RemoteAsrModelCache,
) -> DatasetAsrResponse:
    audio = audio_cache.materialize(request.source)
    output_file = tempfile.NamedTemporaryFile(
        prefix=f"{audio.path.stem}_dataset_asr_",
        suffix=".flac",
        delete=False,
    )
    output_path = Path(output_file.name)
    output_file.close()
    try:
        prepared = prepare_audio_transport(
            source_path=audio.path,
            output_path=output_path,
            spec=CANONICAL_MODEL_AUDIO_SPEC,
        )
        results = transcribe_requested_models(
            model_cache=model_cache,
            model_ids=request.models,
            audio=prepare_asr_audio(output_path),
        )
    finally:
        output_path.unlink(missing_ok=True)
    return DatasetAsrResponse(
        audio=materialized_audio(audio),
        prepared_audio_sha256=prepared.sha256,
        prepared_duration_seconds=prepared.duration_seconds,
        results=results,
    )


def materialized_audio(audio: CachedAudioFile) -> MaterializedAudio:
    return MaterializedAudio(
        source=audio.source,
        filename=audio.path.name,
        content_sha256=audio.content_sha256,
        metadata=audio.audio_metadata,
    )
