from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from silero_vad import load_silero_vad
from torch import Tensor

from app.local.analyses.end_of_turn.detectors.livekit_v1_mini import (
    LiveKitEndOfTurnModel,
)
from app.local.analyses.end_of_turn.detectors.pipecat_smart_turn_v3 import (
    SmartTurnV3Inference,
    smart_turn_completion_probability,
)
from app.training.turn_taking.benchmark_models import (
    CandidatePrediction,
    CandidateTargetPoint,
    CompletionCandidatePrediction,
    CompletionTargetPoint,
    SilenceCandidate,
    TurnCompletionCandidate,
)
from app.training.turn_taking.data import load_audio_window

MODEL_SAMPLE_RATE_HZ = 16_000
SILERO_WINDOW_SAMPLES = 512


class AudioPathResolver(Protocol):
    def resolve(self, relative_path: str) -> Path: ...


class FloatProbabilityScorer(Protocol):
    def score(self, audio: NDArray[np.float32]) -> float: ...


class Int16ProbabilityScorer(Protocol):
    def score(self, audio: NDArray[np.int16]) -> float: ...


class CausalSileroModel(Protocol):
    def __call__(self, audio: Tensor, sample_rate_hz: int) -> Tensor: ...

    def reset_states(self) -> None: ...


class CandidateAudioProvider(Protocol):
    def load(
        self,
        candidate: SilenceCandidate | TurnCompletionCandidate,
        start_seconds: float,
        end_seconds: float,
    ) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class RootAudioPathResolver:
    root: Path

    def resolve(self, relative_path: str) -> Path:
        return self.root.joinpath(*PurePosixPath(relative_path).parts)


@dataclass(frozen=True)
class HubAudioPathResolver:
    repository_id: str
    revision: str
    cache_directory: Path | None

    def resolve(self, relative_path: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=self.repository_id,
                repo_type="dataset",
                filename=relative_path,
                revision=self.revision,
                cache_dir=self.cache_directory,
            )
        )


@dataclass(frozen=True)
class ResolvedCandidateAudioProvider:
    path_resolver: AudioPathResolver
    sample_rate_hz: int = MODEL_SAMPLE_RATE_HZ

    def load(
        self,
        candidate: SilenceCandidate | TurnCompletionCandidate,
        start_seconds: float,
        end_seconds: float,
    ) -> NDArray[np.float32]:
        path = self.path_resolver.resolve(candidate.user_audio_path)
        waveform = load_audio_window(
            path=path,
            sample_rate_hz=self.sample_rate_hz,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        return waveform.numpy().astype(np.float32, copy=False)


@dataclass(frozen=True)
class SmartTurnInferenceScorer:
    inference: SmartTurnV3Inference

    def score(self, audio: NDArray[np.float32]) -> float:
        return smart_turn_completion_probability(inference=self.inference, audio=audio)


@dataclass(frozen=True)
class LiveKitInferenceScorer:
    model: LiveKitEndOfTurnModel

    def score(self, audio: NDArray[np.int16]) -> float:
        return self.model.predict(pcm=audio)


def smart_turn_candidate_predictions(
    candidates: tuple[SilenceCandidate, ...],
    audio_provider: CandidateAudioProvider,
    scorer: FloatProbabilityScorer,
    candidate_silence_seconds: float,
    maximum_window_seconds: float,
) -> tuple[CandidatePrediction, ...]:
    predictions: list[CandidatePrediction] = []
    for candidate in candidates:
        target_point = _target_point_at_gate(candidate, candidate_silence_seconds)
        if target_point is None:
            continue
        start_seconds = max(
            candidate.preceding_speech_start_seconds,
            target_point.absolute_time_seconds - maximum_window_seconds,
        )
        audio = audio_provider.load(
            candidate=candidate,
            start_seconds=start_seconds,
            end_seconds=target_point.absolute_time_seconds,
        )
        started_at = time.perf_counter()
        probability = scorer.score(audio)
        inference_duration_seconds = time.perf_counter() - started_at
        predictions.append(
            _prediction(
                candidate=candidate,
                target_point=target_point,
                probability=probability,
                inference_duration_seconds=inference_duration_seconds,
            )
        )
    return tuple(predictions)


def livekit_candidate_predictions(
    candidates: tuple[SilenceCandidate, ...],
    audio_provider: CandidateAudioProvider,
    scorer: Int16ProbabilityScorer,
    candidate_silence_seconds: float,
    maximum_window_seconds: float,
) -> tuple[CandidatePrediction, ...]:
    predictions: list[CandidatePrediction] = []
    for candidate in candidates:
        target_point = _target_point_at_gate(candidate, candidate_silence_seconds)
        if target_point is None:
            continue
        start_seconds = max(0.0, target_point.absolute_time_seconds - maximum_window_seconds)
        float_audio = audio_provider.load(
            candidate=candidate,
            start_seconds=start_seconds,
            end_seconds=target_point.absolute_time_seconds,
        )
        pcm_audio = np.clip(float_audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
        started_at = time.perf_counter()
        probability = scorer.score(pcm_audio)
        inference_duration_seconds = time.perf_counter() - started_at
        predictions.append(
            _prediction(
                candidate=candidate,
                target_point=target_point,
                probability=probability,
                inference_duration_seconds=inference_duration_seconds,
            )
        )
    return tuple(predictions)


def silero_candidate_predictions(
    candidates: tuple[SilenceCandidate, ...],
    audio_provider: CandidateAudioProvider,
    model: CausalSileroModel,
) -> tuple[CandidatePrediction, ...]:
    predictions: list[CandidatePrediction] = []
    for candidate in candidates:
        model.reset_states()
        audio = audio_provider.load(
            candidate=candidate,
            start_seconds=candidate.preceding_speech_start_seconds,
            end_seconds=candidate.end_seconds,
        )
        for start_index in range(0, audio.size, SILERO_WINDOW_SAMPLES):
            window = audio[start_index : start_index + SILERO_WINDOW_SAMPLES]
            if window.size < SILERO_WINDOW_SAMPLES:
                window = np.pad(window, (0, SILERO_WINDOW_SAMPLES - window.size))
            started_at = time.perf_counter()
            speech_probability = float(
                model(torch.from_numpy(window.astype(np.float32, copy=False)), MODEL_SAMPLE_RATE_HZ)
                .reshape(-1)[0]
                .item()
            )
            inference_duration_seconds = time.perf_counter() - started_at
            frame_end_seconds = (
                candidate.preceding_speech_start_seconds
                + (start_index + SILERO_WINDOW_SAMPLES) / MODEL_SAMPLE_RATE_HZ
            )
            silence_duration_seconds = frame_end_seconds - candidate.start_seconds
            if silence_duration_seconds <= 0.0:
                continue
            if frame_end_seconds > candidate.end_seconds + 1e-6:
                continue
            predictions.append(
                CandidatePrediction(
                    candidate_id=candidate.candidate_id,
                    absolute_time_seconds=frame_end_seconds,
                    silence_duration_seconds=silence_duration_seconds,
                    yield_probability=1.0 - speech_probability,
                    inference_duration_seconds=inference_duration_seconds,
                )
            )
    return tuple(predictions)


def load_causal_silero_model(use_onnx: bool) -> CausalSileroModel:
    return cast(CausalSileroModel, load_silero_vad(onnx=use_onnx))


def smart_turn_completion_candidate_predictions(
    candidates: tuple[TurnCompletionCandidate, ...],
    audio_provider: CandidateAudioProvider,
    scorer: FloatProbabilityScorer,
    candidate_silence_seconds: float,
    maximum_window_seconds: float,
) -> tuple[CompletionCandidatePrediction, ...]:
    predictions: list[CompletionCandidatePrediction] = []
    for candidate in candidates:
        target_point = _completion_target_point_at_gate(candidate, candidate_silence_seconds)
        if target_point is None:
            continue
        start_seconds = max(
            candidate.preceding_speech_start_seconds,
            target_point.absolute_time_seconds - maximum_window_seconds,
        )
        audio = audio_provider.load(candidate, start_seconds, target_point.absolute_time_seconds)
        started_at = time.perf_counter()
        probability = scorer.score(audio)
        predictions.append(
            CompletionCandidatePrediction(
                candidate_id=candidate.candidate_id,
                absolute_time_seconds=target_point.absolute_time_seconds,
                elapsed_seconds=target_point.elapsed_seconds,
                completion_probability=probability,
                inference_duration_seconds=time.perf_counter() - started_at,
            )
        )
    return tuple(predictions)


def livekit_completion_candidate_predictions(
    candidates: tuple[TurnCompletionCandidate, ...],
    audio_provider: CandidateAudioProvider,
    scorer: Int16ProbabilityScorer,
    candidate_silence_seconds: float,
    maximum_window_seconds: float,
) -> tuple[CompletionCandidatePrediction, ...]:
    predictions: list[CompletionCandidatePrediction] = []
    for candidate in candidates:
        target_point = _completion_target_point_at_gate(candidate, candidate_silence_seconds)
        if target_point is None:
            continue
        start_seconds = max(0.0, target_point.absolute_time_seconds - maximum_window_seconds)
        float_audio = audio_provider.load(
            candidate,
            start_seconds,
            target_point.absolute_time_seconds,
        )
        pcm_audio = np.clip(float_audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
        started_at = time.perf_counter()
        probability = scorer.score(pcm_audio)
        predictions.append(
            CompletionCandidatePrediction(
                candidate_id=candidate.candidate_id,
                absolute_time_seconds=target_point.absolute_time_seconds,
                elapsed_seconds=target_point.elapsed_seconds,
                completion_probability=probability,
                inference_duration_seconds=time.perf_counter() - started_at,
            )
        )
    return tuple(predictions)


def silero_completion_candidate_predictions(
    candidates: tuple[TurnCompletionCandidate, ...],
    audio_provider: CandidateAudioProvider,
    model: CausalSileroModel,
) -> tuple[CompletionCandidatePrediction, ...]:
    predictions: list[CompletionCandidatePrediction] = []
    for candidate in candidates:
        model.reset_states()
        audio = audio_provider.load(
            candidate,
            candidate.preceding_speech_start_seconds,
            candidate.end_seconds,
        )
        for start_index in range(0, audio.size, SILERO_WINDOW_SAMPLES):
            window = audio[start_index : start_index + SILERO_WINDOW_SAMPLES]
            if window.size < SILERO_WINDOW_SAMPLES:
                window = np.pad(window, (0, SILERO_WINDOW_SAMPLES - window.size))
            started_at = time.perf_counter()
            speech_probability = float(
                model(torch.from_numpy(window.astype(np.float32, copy=False)), MODEL_SAMPLE_RATE_HZ)
                .reshape(-1)[0]
                .item()
            )
            inference_duration = time.perf_counter() - started_at
            absolute_time = (
                candidate.preceding_speech_start_seconds
                + (start_index + SILERO_WINDOW_SAMPLES) / MODEL_SAMPLE_RATE_HZ
            )
            elapsed = absolute_time - candidate.anchor_seconds
            if elapsed <= 0.0 or absolute_time > candidate.end_seconds + 1e-6:
                continue
            predictions.append(
                CompletionCandidatePrediction(
                    candidate_id=candidate.candidate_id,
                    absolute_time_seconds=absolute_time,
                    elapsed_seconds=elapsed,
                    completion_probability=1.0 - speech_probability,
                    inference_duration_seconds=inference_duration,
                )
            )
    return tuple(predictions)


def _completion_target_point_at_gate(
    candidate: TurnCompletionCandidate,
    candidate_silence_seconds: float,
) -> CompletionTargetPoint | None:
    if candidate_silence_seconds <= 0.0:
        raise ValueError("candidate_silence_seconds must be positive.")
    return next(
        (
            target_point
            for target_point in candidate.target_points
            if target_point.elapsed_seconds + 1e-6 >= candidate_silence_seconds
        ),
        None,
    )


def _target_point_at_gate(
    candidate: SilenceCandidate,
    candidate_silence_seconds: float,
) -> CandidateTargetPoint | None:
    if candidate_silence_seconds <= 0.0:
        raise ValueError("candidate_silence_seconds must be positive.")
    return next(
        (
            target_point
            for target_point in candidate.target_points
            if target_point.silence_duration_seconds + 1e-6 >= candidate_silence_seconds
        ),
        None,
    )


def _prediction(
    candidate: SilenceCandidate,
    target_point: CandidateTargetPoint,
    probability: float,
    inference_duration_seconds: float,
) -> CandidatePrediction:
    return CandidatePrediction(
        candidate_id=candidate.candidate_id,
        absolute_time_seconds=target_point.absolute_time_seconds,
        silence_duration_seconds=target_point.silence_duration_seconds,
        yield_probability=probability,
        inference_duration_seconds=inference_duration_seconds,
    )
