from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.local.analyses.end_of_turn.base import EndOfTurnDetectorMode
from app.local.analyses.end_of_turn.registry import (
    available_detectors,
    run_all_detectors,
    run_detectors,
)
from app.local.analyses.end_of_turn.service import analysis_to_json, analyze_session_audio
from app.local.data.sessions import SpeakerName, session_audio_path, session_metadata_path
from app.local.data.transcripts import read_transcript_turns

router = APIRouter(prefix="/api/end-of-turn", tags=["end-of-turn"])


@router.get("/detectors")
def list_detector_modes() -> dict[str, object]:
    return {
        "detectors": [
            {
                "mode": detector_info.mode.value,
                "label": detector_info.label,
                "description": detector_info.description,
            }
            for detector_info in available_detectors()
        ]
    }


@router.get("/analyze")
def analyze_end_of_turn(
    identifier: str = Query(alias="id"),
    detectors: str | None = None,
) -> dict[str, object]:
    try:
        selected_detector_modes = parse_selected_detector_modes(detectors=detectors)
        speaker1_path = session_audio_path(
            identifier=identifier,
            speaker_name=SpeakerName.SPEAKER1,
        )
        speaker2_path = session_audio_path(
            identifier=identifier,
            speaker_name=SpeakerName.SPEAKER2,
        )
        transcript_turns = read_transcript_turns(
            metadata_path=session_metadata_path(identifier=identifier)
        )
        if selected_detector_modes is None:
            baseline_results = run_all_detectors(speaker1_path=speaker1_path)
        else:
            baseline_results = run_detectors(
                speaker1_path=speaker1_path,
                selected_modes=selected_detector_modes,
            )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    audio_analysis = analyze_session_audio(
        speaker1_path=speaker1_path,
        speaker2_path=speaker2_path,
        transcript_turns=transcript_turns,
        baseline_results=baseline_results,
    )
    return {
        "session_id": identifier,
        "speaker1_audio_url": f"/api/audio/{identifier}/{SpeakerName.SPEAKER1.value}",
        "speaker2_audio_url": f"/api/audio/{identifier}/{SpeakerName.SPEAKER2.value}",
        "analysis": analysis_to_json(audio_analysis=audio_analysis),
    }


def parse_selected_detector_modes(
    detectors: str | None,
) -> list[EndOfTurnDetectorMode] | None:
    if detectors is None:
        return None
    if detectors == "":
        return []

    selected_modes: list[EndOfTurnDetectorMode] = []
    for detector_mode in detectors.split(","):
        try:
            selected_modes.append(EndOfTurnDetectorMode(detector_mode))
        except ValueError as error:
            raise ValueError(f"Unknown end-of-turn detector mode: {detector_mode}") from error
    return selected_modes
