import wave
from pathlib import Path

import pytest

from app.local.analyses.asr.models import (
    AsrAnalysisResponse,
    AsrModelInfo,
    AsrModelMode,
    AsrModelRun,
)
from app.local.analyses.asr.service import IngestedAsrTranscriptSource
from app.local.analyses.end_of_turn.detectors import asr_two_speaker_annotation as asr_annotation
from app.local.analyses.end_of_turn.detectors.asr_two_speaker_annotation import (
    ASR_MODEL_MODE,
    asr_two_speaker_annotation_detector,
    asr_two_speaker_annotation_detectors,
    merged_asr_turns,
)
from app.local.asr.service import AsrTranscriptCache, RemoteAsrClientFactory
from app.local.asr.transcript import SpeakerTrack, TranscriptionResult, Word


def test_merged_asr_turns_groups_each_speaker_and_orders_channels_by_time() -> None:
    turns = merged_asr_turns(
        speaker1_words=(
            Word(text="Hello", start_seconds=1.0, end_seconds=1.3),
            Word(text="there", start_seconds=1.5, end_seconds=1.8),
            Word(text="Later", start_seconds=4.0, end_seconds=4.4),
        ),
        speaker2_words=(
            Word(text="yeah", start_seconds=2.0, end_seconds=2.3),
            Word(text="continuing", start_seconds=2.4, end_seconds=2.9),
        ),
        turn_gap_seconds=1.5,
    )

    assert [(turn.speaker, turn.text) for turn in turns] == [
        ("Speaker1", "Hello there"),
        ("Speaker2", "yeah continuing"),
        ("Speaker1", "Later"),
    ]


@pytest.mark.parametrize(
    "word",
    [
        Word(text="missing-start", end_seconds=1.0),
        Word(text="missing-end", start_seconds=1.0),
        Word(text="reversed", start_seconds=2.0, end_seconds=1.0),
    ],
)
def test_merged_asr_turns_rejects_invalid_word_timestamps(word: Word) -> None:
    with pytest.raises(ValueError, match="ASR word"):
        merged_asr_turns(
            speaker1_words=(word,),
            speaker2_words=(),
            turn_gap_seconds=1.5,
        )


def test_detector_requests_cached_union_for_both_speakers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_tracks: list[SpeakerTrack] = []

    def fake_analyze_asr(
        session_id: str,
        speaker_track: SpeakerTrack,
        selected_models: tuple[AsrModelMode, ...],
        reference_words: tuple[Word, ...],
        cache: AsrTranscriptCache,
        ingested_transcript_source: IngestedAsrTranscriptSource,
        remote_client_factory: RemoteAsrClientFactory,
    ) -> AsrAnalysisResponse:
        del reference_words, cache, ingested_transcript_source, remote_client_factory
        assert session_id == "session"
        assert selected_models == (ASR_MODEL_MODE,)
        requested_tracks.append(speaker_track)
        words = (
            (
                Word(text="first", start_seconds=0.5, end_seconds=1.0),
                Word(text="second", start_seconds=3.0, end_seconds=3.5),
            )
            if speaker_track == SpeakerTrack.SPEAKER1
            else (Word(text="reply", start_seconds=0.8, end_seconds=2.0),)
        )
        return _asr_response(speaker_track=speaker_track, words=words)

    monkeypatch.setattr(asr_annotation, "analyze_asr", fake_analyze_asr)
    speaker1_path = tmp_path / "session_speaker1.wav"
    speaker2_path = tmp_path / "session_speaker2.wav"
    _write_silent_wave(path=speaker1_path, duration_seconds=5.0)
    _write_silent_wave(path=speaker2_path, duration_seconds=5.0)

    result = asr_two_speaker_annotation_detector().analyze(speaker1_path=speaker1_path)

    assert requested_tracks == [SpeakerTrack.SPEAKER1, SpeakerTrack.SPEAKER2]
    assert result.name == "asr_two_speaker_annotation"
    assert [event.time_seconds for event in result.end_of_turn_events] == [1.0, 3.5]
    assert result.backchannel_spans == []


def test_paired_detectors_share_transcripts_and_mark_speaker2_interruption_onset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_tracks: list[SpeakerTrack] = []

    def fake_analyze_asr(
        session_id: str,
        speaker_track: SpeakerTrack,
        selected_models: tuple[AsrModelMode, ...],
        reference_words: tuple[Word, ...],
        cache: AsrTranscriptCache,
        ingested_transcript_source: IngestedAsrTranscriptSource,
        remote_client_factory: RemoteAsrClientFactory,
    ) -> AsrAnalysisResponse:
        del (
            session_id,
            selected_models,
            reference_words,
            cache,
            ingested_transcript_source,
            remote_client_factory,
        )
        requested_tracks.append(speaker_track)
        words = (
            (Word(text="speaking", start_seconds=0.2, end_seconds=1.5),)
            if speaker_track == SpeakerTrack.SPEAKER1
            else (Word(text="interrupting", start_seconds=0.8, end_seconds=2.0),)
        )
        return _asr_response(speaker_track=speaker_track, words=words)

    monkeypatch.setattr(asr_annotation, "analyze_asr", fake_analyze_asr)
    speaker1_path = tmp_path / "session_speaker1.wav"
    speaker2_path = tmp_path / "session_speaker2.wav"
    _write_silent_wave(path=speaker1_path, duration_seconds=5.0)
    _write_silent_wave(path=speaker2_path, duration_seconds=5.0)
    speaker1_detector, speaker2_detector = asr_two_speaker_annotation_detectors()

    speaker1_detector.analyze(speaker1_path=speaker1_path)
    speaker2_result = speaker2_detector.analyze(speaker1_path=speaker1_path)

    assert requested_tracks == [SpeakerTrack.SPEAKER1, SpeakerTrack.SPEAKER2]
    assert speaker2_result.name == "asr_two_speaker_annotation_speaker2"
    assert len(speaker2_result.interruption_events) == 1
    interruption = speaker2_result.interruption_events[0]
    assert interruption.time_seconds == 0.8


def _asr_response(
    speaker_track: SpeakerTrack,
    words: tuple[Word, ...],
) -> AsrAnalysisResponse:
    model_info = AsrModelInfo(
        mode=ASR_MODEL_MODE,
        label="Test union",
        description="Test union transcript.",
    )
    transcription = TranscriptionResult(
        model_name=ASR_MODEL_MODE.value,
        audio_path="test.wav",
        track=speaker_track,
        audio_duration_seconds=5.0,
        processing_time_seconds=0.0,
        words=words,
        model_identifier="test",
    )
    return AsrAnalysisResponse(
        session_id="session",
        speaker_track=speaker_track,
        audio_url="/test.wav",
        analyzed_duration_seconds=5.0,
        reference_word_count=0,
        runs=(AsrModelRun(model=model_info, transcription=transcription, metrics=None),),
    )


def _write_silent_wave(path: Path, duration_seconds: float) -> None:
    sample_rate = 16_000
    sample_count = round(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(bytes(sample_count * 2))
