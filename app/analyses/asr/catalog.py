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
            description="Timing union filtered by rolling loudness and other-channel dominance.",
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
                "Merged consensus filtered by rolling loudness and other-channel dominance."
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


def requires_merged_consensus(selected_models: tuple[AsrModelMode, ...]) -> bool:
    return any(
        model_mode
        in {
            AsrModelMode.MERGED_CONSENSUS,
            AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED,
        }
        for model_mode in selected_models
    )


def requires_parakeet_canary_union(selected_models: tuple[AsrModelMode, ...]) -> bool:
    return any(
        model_mode
        in {
            AsrModelMode.PARAKEET_CANARY_CONSENSUS,
            AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED,
        }
        for model_mode in selected_models
    )


def is_filtered_model(model_mode: AsrModelMode) -> bool:
    return model_mode in FILTERED_ASR_MODEL_MODES


def source_mode_for_filtered_model(model_mode: AsrModelMode) -> AsrModelMode:
    match model_mode:
        case AsrModelMode.PARAKEET_TDT_CROSSTALK_FILTERED:
            return AsrModelMode.PARAKEET_TDT
        case AsrModelMode.WHISPERX_CROSSTALK_FILTERED:
            return AsrModelMode.WHISPERX
        case AsrModelMode.CANARY_CROSSTALK_FILTERED:
            return AsrModelMode.CANARY
        case AsrModelMode.NEMOTRON_3_5_CROSSTALK_FILTERED:
            return AsrModelMode.NEMOTRON_3_5
        case AsrModelMode.PARAKEET_CANARY_CROSSTALK_FILTERED:
            return AsrModelMode.PARAKEET_CANARY_CONSENSUS
        case AsrModelMode.MERGED_CONSENSUS_CROSSTALK_FILTERED:
            return AsrModelMode.MERGED_CONSENSUS
        case _:
            raise ValueError(f"ASR model mode is not crosstalk filtered: {model_mode.value}")
