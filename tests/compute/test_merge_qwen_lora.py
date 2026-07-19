from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_ADAPTER_NAME,
    LANGUAGE_MODEL_ADAPTER_REVISION,
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
)
from deployment.compute import merge_qwen_lora


class FakeMergedModel:
    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        self.dtype = dtype
        self.save_arguments: tuple[Path, bool, str] | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"model.weight": torch.ones((1,), dtype=self.dtype)}

    def save_pretrained(
        self,
        output_directory: Path,
        *,
        safe_serialization: bool,
        max_shard_size: str,
    ) -> None:
        self.save_arguments = (output_directory, safe_serialization, max_shard_size)
        (output_directory / "model.safetensors").write_bytes(b"test checkpoint")


class FakeAdapterModel:
    def __init__(self, merged_model: FakeMergedModel) -> None:
        self.merged_model = merged_model
        self.merge_arguments: tuple[bool, bool] | None = None

    def merge_and_unload(
        self,
        *,
        progressbar: bool,
        safe_merge: bool,
    ) -> FakeMergedModel:
        self.merge_arguments = (progressbar, safe_merge)
        return self.merged_model


class FakeTokenizer:
    def __init__(self) -> None:
        self.output_directory: Path | None = None

    def save_pretrained(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        (output_directory / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_merge_builds_pinned_standalone_hugging_face_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    merged_model = FakeMergedModel()
    adapter_model = FakeAdapterModel(merged_model)
    tokenizer = FakeTokenizer()
    loaded_base_model: object = object()
    base_load: tuple[str, str, torch.dtype, str, bool] | None = None
    adapter_load: tuple[object, str, str, bool] | None = None
    tokenizer_load: tuple[str, str] | None = None

    def load_base_model(
        model_name: str,
        *,
        revision: str,
        dtype: torch.dtype,
        device_map: str,
        low_cpu_mem_usage: bool,
    ) -> object:
        nonlocal base_load
        base_load = (model_name, revision, dtype, device_map, low_cpu_mem_usage)
        return loaded_base_model

    def load_adapter(
        base_model: object,
        adapter_name: str,
        *,
        revision: str,
        is_trainable: bool,
    ) -> FakeAdapterModel:
        nonlocal adapter_load
        adapter_load = (base_model, adapter_name, revision, is_trainable)
        return adapter_model

    def load_tokenizer(
        model_name: str,
        *,
        revision: str,
    ) -> FakeTokenizer:
        nonlocal tokenizer_load
        tokenizer_load = (model_name, revision)
        return tokenizer

    monkeypatch.setattr(
        merge_qwen_lora.AutoModelForCausalLM,
        "from_pretrained",
        load_base_model,
    )
    monkeypatch.setattr(
        merge_qwen_lora.PeftModelForCausalLM,
        "from_pretrained",
        load_adapter,
    )
    monkeypatch.setattr(
        merge_qwen_lora.AutoTokenizer,
        "from_pretrained",
        load_tokenizer,
    )

    output_directory = tmp_path / "merged"
    provenance = merge_qwen_lora.merge_qwen_lora(
        merge_qwen_lora.MergeOptions(
            output_directory=output_directory,
            destination_repository_id="BertilBraun/qwen3-1.7b-voice-light-tool-use-merged",
        )
    )

    assert base_load == (
        LANGUAGE_MODEL_NAME,
        LANGUAGE_MODEL_REVISION,
        torch.bfloat16,
        "cpu",
        True,
    )
    assert adapter_load == (
        loaded_base_model,
        LANGUAGE_MODEL_ADAPTER_NAME,
        LANGUAGE_MODEL_ADAPTER_REVISION,
        False,
    )
    assert tokenizer_load == (LANGUAGE_MODEL_NAME, LANGUAGE_MODEL_REVISION)
    assert adapter_model.merge_arguments == (True, True)
    assert merged_model.save_arguments == (output_directory, True, "5GB")
    assert tokenizer.output_directory == output_directory
    assert provenance.base_model.revision == LANGUAGE_MODEL_REVISION
    assert provenance.adapter.revision == LANGUAGE_MODEL_ADAPTER_REVISION
    assert provenance.merge.dtype == "bfloat16"
    assert provenance.merge.device == "cpu"
    assert provenance.merge.maximum_shard_size == "5GB"
    assert (output_directory / merge_qwen_lora.PROVENANCE_FILENAME).is_file()
    model_card = (output_directory / merge_qwen_lora.MODEL_CARD_FILENAME).read_text(
        encoding="utf-8"
    )
    assert f"base_model: {LANGUAGE_MODEL_NAME}" in model_card
    assert LANGUAGE_MODEL_REVISION in model_card
    assert LANGUAGE_MODEL_ADAPTER_REVISION in model_card
    assert "does not\nrequire PEFT or dynamic LoRA loading" in model_card


def test_merge_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output_directory = tmp_path / "merged"
    output_directory.mkdir()
    (output_directory / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        merge_qwen_lora.prepare_output_directory(output_directory)


def test_merge_rejects_non_bfloat16_floating_weights() -> None:
    with pytest.raises(RuntimeError, match="non-BF16"):
        merge_qwen_lora.validate_bfloat16_checkpoint(FakeMergedModel(torch.float32))


@pytest.mark.parametrize(
    "repository_id",
    ["model", "/model", "owner/", "owner/model/extra", "owner/model name"],
)
def test_merge_cli_rejects_invalid_destination_repository(
    repository_id: str,
) -> None:
    with pytest.raises(SystemExit):
        merge_qwen_lora.parse_options(
            [
                "--output-directory",
                "merged",
                "--destination-repository-id",
                repository_id,
            ]
        )
