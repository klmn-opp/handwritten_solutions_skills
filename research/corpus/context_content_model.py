"""Deterministic whole-word-group ink decoder, shared by training and generation."""

import torch
import torch.nn.functional as F
from torch import nn

from research.corpus.reference_generator import block


class ContentInk(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec = {
            "name": "aligned-content-ink/1",
            "height": 64,
            "width": 512,
            "context": True,
            "stochastic_style": False,
            "math_generation": False,
            "pixel_copy_output": False,
        }
        self.e1, self.e2, self.e3 = block(1, 12, 2), block(12, 24, 2), block(24, 48, 2)
        self.context = nn.Sequential(
            nn.Conv1d(48, 48, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(48, 48, 5, padding=4, dilation=2),
            nn.SiLU(),
            nn.Conv1d(48, 48, 5, padding=8, dilation=4),
        )
        self.d3 = block(48, 48)
        self.d2 = block(48 + 24, 24)
        self.d1 = block(24 + 12, 12)
        self.out = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1), nn.SiLU(), nn.Conv2d(12, 1, 1), nn.Tanh()
        )

    def forward(self, reference):
        a = self.e1(reference)
        b = self.e2(a)
        c = self.e3(b)
        c = c + self.context(c.mean(2))[:, :, None, :]
        x = self.d3(c)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.d2(torch.cat([x, b], 1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.d1(torch.cat([x, a], 1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.out(x)


class FrozenTranscriptionLoss:
    """Frozen local TrOCR content teacher; never an independent evaluator of this learner."""

    def __init__(self, checkpoint: str, device="cuda"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.processor = TrOCRProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(checkpoint, local_files_only=True)
            .to(device)
            .eval()
        )
        self.model.config.decoder_start_token_id = (
            self.model.generation_config.decoder_start_token_id
        )
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Checkpoint lacks decoder start token")
        self.model.requires_grad_(False)
        self.device = device

    def labels(self, texts: list[str]):
        labels = self.processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids.to(
            self.device
        )
        return labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

    def __call__(self, ink, labels):
        # Match local preprocessor: white RGB, bilinear 384 square, mean/std=.5.
        pixels = F.interpolate(
            1 - ink, size=(384, 384), mode="bilinear", align_corners=False
        ).expand(-1, 3, -1, -1)
        return self.model(pixel_values=(pixels - 0.5) / 0.5, labels=labels, use_cache=False).loss
