from __future__ import annotations

from app.analyses.asr.models import AsrModelInfo, AsrModelMode

REMOTE_ASR_MODEL_MODES = (
    AsrModelMode.PARAKEET_TDT,
    AsrModelMode.WHISPERX,
    AsrModelMode.CANARY,
    AsrModelMode.NEMOTRON_3_5,
)

FILTERED_ASR_MODEL_MODES = (
    AsrModelMode.PARAKEET_TDT_CROSSTALK_FILTERED,
    AsrModelMode.WHISPERX_CROSSTALK_FILTERED,
    AsrModelMode.CANARY_CROSSTALK_FILTERED,
    AsrModelMode.NEMOTRON_3_5_CROSSTALK_FILTERED,
    AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED,
    AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED,
)


def available_asr_models() -> tuple[AsrModelInfo, ...]:
    return (
        AsrModelInfo(
            mode=AsrModelMode.PARAKEET_TDT,
            label="Parakeet TDT 0.6B v3",
            description="NVIDIA Parakeet word-timestamp ASR via Transformers.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.PARAKEET_TDT_CROSSTALK_FILTERED,
            label="Parakeet - crosstalk filtered",
            description="Parakeet words filtered by rolling loudness and other-channel dominance.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.WHISPERX,
            label="WhisperX large-v3",
            description="Faster-Whisper transcription plus WhisperX word alignment.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.WHISPERX_CROSSTALK_FILTERED,
            label="WhisperX - crosstalk filtered",
            description="WhisperX words filtered by rolling loudness and other-channel dominance.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.CANARY,
            label="Canary 1B v2",
            description="NVIDIA NeMo Canary with native word timestamp output.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.CANARY_CROSSTALK_FILTERED,
            label="Canary - crosstalk filtered",
            description="Canary words filtered by rolling loudness and other-channel dominance.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.NEMOTRON_3_5,
            label="Nemotron 3.5 ASR streaming 0.6B",
            description="NVIDIA Nemotron transcript aligned with WhisperX word timestamps.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.NEMOTRON_3_5_CROSSTALK_FILTERED,
            label="Nemotron - crosstalk filtered",
            description="Nemotron words filtered by rolling loudness and other-channel dominance.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.PARAKEET_CANARY_CONSENSUS,
            label="Parakeet + Canary timing union",
            description=(
                "All Parakeet words plus Canary-only timestamp coverage, without "
                "overlapping duplicates."
            ),
        ),
        AsrModelInfo(
            mode=AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED,
            label="Parakeet + Canary union - crosstalk filtered",
            description="Union of independently crosstalk-filtered Parakeet and Canary words.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.MERGED_CONSENSUS,
            label="Merged ASR consensus",
            description="Local alignment merge of selected timestamped ASR outputs.",
        ),
        AsrModelInfo(
            mode=AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED,
            label="Merged ASR consensus - crosstalk filtered",
            description=(
                "Consensus of four independently crosstalk-filtered remote model outputs."
            ),
        ),
    )


def is_remote_model(model_mode: AsrModelMode) -> bool:
    return model_mode in REMOTE_ASR_MODEL_MODES


def remote_dependencies_for_model_mode(model_mode: AsrModelMode) -> tuple[AsrModelMode, ...]:
    match model_mode:
        case AsrModelMode.MERGED_CONSENSUS | AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED:
            return REMOTE_ASR_MODEL_MODES
        case (
            AsrModelMode.PARAKEET_CANARY_CONSENSUS | AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED
        ):
            return (AsrModelMode.PARAKEET_TDT, AsrModelMode.CANARY)
        case AsrModelMode.PARAKEET_TDT_CROSSTALK_FILTERED:
            return (AsrModelMode.PARAKEET_TDT,)
        case AsrModelMode.WHISPERX_CROSSTALK_FILTERED:
            return (AsrModelMode.WHISPERX,)
        case AsrModelMode.CANARY_CROSSTALK_FILTERED:
            return (AsrModelMode.CANARY,)
        case AsrModelMode.NEMOTRON_3_5_CROSSTALK_FILTERED:
            return (AsrModelMode.NEMOTRON_3_5,)
        case _:
            return ()


def is_filtered_model(model_mode: AsrModelMode) -> bool:
    return model_mode in FILTERED_ASR_MODEL_MODES


def filtered_mode_for_remote_model(model_mode: AsrModelMode) -> AsrModelMode:
    match model_mode:
        case AsrModelMode.PARAKEET_TDT:
            return AsrModelMode.PARAKEET_TDT_CROSSTALK_FILTERED
        case AsrModelMode.WHISPERX:
            return AsrModelMode.WHISPERX_CROSSTALK_FILTERED
        case AsrModelMode.CANARY:
            return AsrModelMode.CANARY_CROSSTALK_FILTERED
        case AsrModelMode.NEMOTRON_3_5:
            return AsrModelMode.NEMOTRON_3_5_CROSSTALK_FILTERED
        case _:
            raise ValueError(f"ASR model mode is not remote: {model_mode.value}")
