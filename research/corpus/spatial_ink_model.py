"""Local character conditions and one shared writer state; no whole-word text MLP."""

import torch
import torch.nn.functional as F
from torch import nn

from research.corpus.ink_diffusion_model import Residual
from research.corpus.latent_ink_model import LatentInk


class SpatialInk(LatentInk):
    architecture = "spatial-character-shared-writer-ink/1"

    def __init__(self):
        super().__init__()
        self.enc[0] = Residual(24, 32)
        self.style = nn.Parameter(torch.zeros(64))
        self.context.requires_grad_(False)

    def warm_start(self, state: dict, tokens: torch.Tensor) -> None:
        base = LatentInk().to(tokens.device)
        base.load_state_dict(state)
        copied = {k: v.clone() for k, v in state.items()}
        for key in ["enc.0.first.weight", "enc.0.skip.weight"]:
            copied[key] = torch.cat([copied[key], torch.zeros_like(copied[key][:, :8])], 1)
        with torch.no_grad():
            copied["style"] = base.context(base.embedding(tokens).flatten(1)).mean(0)
        self.load_state_dict(copied)

    def local_field(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1:] != (32,) or positions.shape != (*tokens.shape, 2):
            raise ValueError("Exact reference-only character positions required")
        if bool(((tokens != 0).sum(1) == 0).any()):
            raise ValueError("Empty text condition")
        x = (torch.arange(64, device=tokens.device) + 0.5) / 64
        center, width = positions.unbind(-1)
        weights = torch.exp(
            -0.5 * ((x - center[..., None]) / (width[..., None] * 0.65 + 0.015)).square()
        )
        weights = weights * (tokens != 0)[..., None]
        weights = weights / weights.sum(1, keepdim=True).clamp(min=1e-5)
        return torch.bmm(self.embedding(tokens).transpose(1, 2), weights)[:, :, None].expand(
            -1, -1, 16, -1
        )

    def forward(self, noisy, reference, tokens, time, positions):
        if noisy.shape != reference.shape or noisy.shape[1:] != (8, 16, 64):
            raise ValueError("Matching spatial ink required")
        frequencies = torch.exp(torch.arange(16, device=time.device) * (-9.210340371976184 / 15))
        phase = time[:, None] * frequencies[None] * 1000
        condition = torch.cat(
            [
                self.time(torch.cat([phase.sin(), phase.cos()], 1)),
                self.style[None].expand(len(noisy), -1),
            ],
            1,
        )
        h = torch.cat([noisy, reference, self.local_field(tokens, positions)], 1)
        features = []
        for i, layer in enumerate(self.enc):
            h = layer(h if i == 0 else F.avg_pool2d(h, 2), condition)
            features.append(h)
        h = self.mid(h, condition)
        for layer, skip in zip(self.dec, reversed(features[:-1]), strict=True):
            h = layer(
                torch.cat([F.interpolate(h, size=skip.shape[-2:], mode="nearest"), skip], 1),
                condition,
            )
        return self.out(h)
