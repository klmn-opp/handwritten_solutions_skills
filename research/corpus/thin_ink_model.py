"""Experimental real-ink losses and optional local text conditioning; Torch optional."""

import torch
import torch.nn.functional as F
from torch import nn

from research.corpus.latent_ink_model import LatentInk, reconstruction_loss


class LocalLatentInk(LatentInk):
    architecture = "locally-conditioned-spatial-ink-denoiser/2"

    def __init__(self, *, local: bool):
        super().__init__()
        self.local = local
        if local:
            self.local_projection = nn.Conv2d(8, 8, 1, bias=False)
            nn.init.zeros_(self.local_projection.weight)

    def forward(self, noisy, reference, tokens, time, positions):
        if positions.shape != (*tokens.shape, 2):
            raise ValueError("Exact reference-only token positions required")
        if self.local:
            x = (
                torch.arange(reference.shape[-1], device=reference.device) + 0.5
            ) / reference.shape[-1]
            center, width = positions.unbind(-1)
            weights = torch.exp(
                -0.5 * ((x - center[..., None]) / (width[..., None] * 0.75 + 0.02)).square()
            )
            weights = weights * (tokens != 0)[..., None]
            weights = weights / weights.sum(1, keepdim=True).clamp(min=1e-6)
            field = torch.bmm(self.embedding(tokens).transpose(1, 2), weights)
            field = field[:, :, None].expand(-1, -1, reference.shape[-2], -1)
            reference = reference + self.local_projection(field)
        return super().forward(noisy, reference, tokens, time)


def ink_detail_loss(logits, target, weights):
    """Retain small real components and signed edges; never force a font target."""
    if logits.shape != target.shape or weights.shape != target.shape:
        raise ValueError("Matching real target/weight/logit shapes required")
    base = reconstruction_loss(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    small = (bce * weights).sum((1, 2, 3)) / weights.sum((1, 2, 3)).clamp(min=1)
    p = logits.sigmoid()
    dx = (p[..., 1:] - p[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (p[..., 1:, :] - p[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    edge = dx.abs().mean() + dy.abs().mean()
    return base + 0.25 * small.mean() + 2 * edge
