from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from transformers import NemotronAsrStreamingEncoder, NemotronAsrStreamingProcessor

from app.training.turn_taking.model import freeze_module


@dataclass(frozen=True)
class BackboneFeatures:
    taps: tuple[Tensor, ...]
    frame_mask: Tensor


class FeatureBackbone(Protocol):
    def extract(self, waveforms: Tensor, waveform_lengths: Tensor) -> BackboneFeatures: ...


class NemotronStreamingBackbone(nn.Module):
    def __init__(
        self,
        model_identifier: str,
        tap_layer_indices: tuple[int, ...],
        lookahead_tokens: int,
    ) -> None:
        super().__init__()
        self.processor = NemotronAsrStreamingProcessor.from_pretrained(model_identifier)
        self.encoder = NemotronAsrStreamingEncoder.from_pretrained(model_identifier)
        freeze_module(self.encoder)
        self.tap_layer_indices = tap_layer_indices
        self.lookahead_tokens = lookahead_tokens

    def extract(self, waveforms: Tensor, waveform_lengths: Tensor) -> BackboneFeatures:
        audio = [
            waveform[:length].detach().cpu().numpy()
            for waveform, length in zip(waveforms, waveform_lengths, strict=True)
        ]
        processed = self.processor(
            audio,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        device = self.encoder.device
        input_features = processed.input_features.to(device=device, dtype=self.encoder.dtype)
        attention_mask = processed.attention_mask.to(device=device)
        with torch.no_grad():
            output = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                num_lookahead_tokens=self.lookahead_tokens,
                output_hidden_states=True,
                return_dict=True,
            )
        if output.hidden_states is None:
            raise ValueError("Nemotron encoder did not return hidden states.")
        if output.attention_mask is None:
            raise ValueError("Nemotron encoder did not return an attention mask.")
        taps = tuple(output.hidden_states[index].detach() for index in self.tap_layer_indices)
        return BackboneFeatures(taps=taps, frame_mask=output.attention_mask.bool())
