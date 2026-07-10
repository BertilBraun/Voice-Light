from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.analyses.end_of_turn.base import EndOfTurnDetectorMode
from app.analyses.end_of_turn.registry import available_detectors, run_detector
from app.analyses.end_of_turn.service import analysis_to_json, analyze_session_audio
from app.data.sessions import SpeakerName, session_audio_path

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
    mode: EndOfTurnDetectorMode = EndOfTurnDetectorMode.NAIVE_VAD_FLOOR,
) -> dict[str, object]:
    try:
        speaker1_path = session_audio_path(
            identifier=identifier,
            speaker_name=SpeakerName.SPEAKER1,
        )
        speaker2_path = session_audio_path(
            identifier=identifier,
            speaker_name=SpeakerName.SPEAKER2,
        )
        baseline_result = run_detector(mode=mode, speaker1_path=speaker1_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    audio_analysis = analyze_session_audio(
        speaker1_path=speaker1_path,
        speaker2_path=speaker2_path,
        baseline_results=[baseline_result],
    )
    return {
        "session_id": identifier,
        "speaker1_audio_url": f"/api/audio/{identifier}/{SpeakerName.SPEAKER1.value}",
        "speaker2_audio_url": f"/api/audio/{identifier}/{SpeakerName.SPEAKER2.value}",
        "analysis": analysis_to_json(audio_analysis=audio_analysis),
    }
