from __future__ import annotations

import tempfile
from pathlib import Path

from app.compute.asr.models.base import prepare_asr_audio
from app.compute.asr.service import RemoteAsrModelCache, transcribe_requested_models
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.audio import load_audio, pad_audio_tracks_to_shared_timeline
from app.shared.audio.crosstalk import (
    CrosstalkFilterConfig,
    crosstalk_frame_power,
    filter_crosstalk_words,
    parakeet_canary_union_words,
)
from app.shared.audio.s3 import CachedAudioFile, S3AudioCache
from app.shared.audio.transport import CANONICAL_MODEL_AUDIO_SPEC, prepare_audio_transport
from app.shared.compute_api import (
    DatasetAsrRequest,
    DatasetAsrResponse,
    DatasetLanguageRequest,
    DatasetLanguageResponse,
    DatasetLanguageTrackResponse,
    DatasetQualityRequest,
    DatasetQualityResponse,
    DatasetTrackTranscripts,
    MaterializedAudio,
)
from app.shared.language import (
    active_probe_windows,
    prepare_language_probe_audio,
    track_language_status,
)
from app.shared.quality_analysis.preprocessing import prepare_audio_track
from app.shared.quality_analysis.service import score_two_track_sample
from app.shared.quality_analysis.vad import VadConfig, detect_speech_segments_pair
from app.shared.storage.local import LocalStorageBackend

DATASET_VAD_VERSION = "energy-pair-v1"
CROSSTALK_CONFIG = CrosstalkFilterConfig()


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


def analyze_dataset_quality(
    request: DatasetQualityRequest,
    audio_cache: S3AudioCache,
) -> DatasetQualityResponse:
    speaker1_file = audio_cache.materialize(request.speaker1)
    speaker2_file = audio_cache.materialize(request.speaker2)
    storage = LocalStorageBackend()
    shared_audio = pad_audio_tracks_to_shared_timeline(
        speaker1=load_audio(
            storage=storage,
            path=speaker1_file.path.as_posix(),
            target_sample_rate=16_000,
        ),
        speaker2=load_audio(
            storage=storage,
            path=speaker2_file.path.as_posix(),
            target_sample_rate=16_000,
        ),
    )
    prepared_speaker1 = prepare_audio_track(shared_audio.speaker1)
    prepared_speaker2 = prepare_audio_track(shared_audio.speaker2)
    speaker1_vad, speaker2_vad = detect_speech_segments_pair(
        speaker1_samples=prepared_speaker1.samples,
        speaker2_samples=prepared_speaker2.samples,
        sample_rate=prepared_speaker1.metadata.sample_rate,
        config=VadConfig(),
    )
    speaker1_words = track_union_words(request.speaker1_transcripts)
    speaker2_words = track_union_words(request.speaker2_transcripts)
    speaker1_power = crosstalk_frame_power(
        target_audio=shared_audio.speaker1,
        other_audio=shared_audio.speaker2,
        config=CROSSTALK_CONFIG,
    )
    speaker2_power = crosstalk_frame_power(
        target_audio=shared_audio.speaker2,
        other_audio=shared_audio.speaker1,
        config=CROSSTALK_CONFIG,
    )
    filtered_speaker1_words = filter_crosstalk_words(
        words=speaker1_words,
        frame_power_pair=speaker1_power,
        config=CROSSTALK_CONFIG,
    )
    filtered_speaker2_words = filter_crosstalk_words(
        words=speaker2_words,
        frame_power_pair=speaker2_power,
        config=CROSSTALK_CONFIG,
    )
    quality_result = score_two_track_sample(
        sample_id=request.sample_id,
        speaker1_audio=shared_audio.speaker1,
        speaker2_audio=shared_audio.speaker2,
        speaker1_uri=request.speaker1.uri,
        speaker2_uri=request.speaker2.uri,
        precomputed_vad=(speaker1_vad, speaker2_vad),
    )
    return DatasetQualityResponse(
        speaker1_audio=materialized_audio(speaker1_file),
        speaker2_audio=materialized_audio(speaker2_file),
        vad_version=DATASET_VAD_VERSION,
        speaker1_vad=speaker1_vad,
        speaker2_vad=speaker2_vad,
        speaker1_filtered_words=filtered_speaker1_words,
        speaker2_filtered_words=filtered_speaker2_words,
        quality_result=quality_result,
    )


def track_union_words(
    transcripts: DatasetTrackTranscripts,
) -> tuple[TimestampedWord, ...]:
    if transcripts.parakeet.model_id is not AsrModelId.PARAKEET_TDT:
        raise ValueError("Parakeet transcript has the wrong model identifier.")
    if transcripts.canary.model_id is not AsrModelId.CANARY:
        raise ValueError("Canary transcript has the wrong model identifier.")
    return parakeet_canary_union_words(
        parakeet_words=transcripts.parakeet.words,
        canary_words=transcripts.canary.words,
    )


def materialized_audio(audio: CachedAudioFile) -> MaterializedAudio:
    return MaterializedAudio(
        source=audio.source,
        filename=audio.path.name,
        content_sha256=audio.content_sha256,
        metadata=audio.audio_metadata,
    )
