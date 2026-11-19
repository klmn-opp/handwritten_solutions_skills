"""Learned contextual contour corrections. Font topology remains a strong prior.

This constrained offline-image component is not free-form line generation, a
trajectory model, or random geometric jitter. Source pixels are never modified.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ContourInk(nn.Module):
    def __init__(self, context: bool = True, bound: float = 2.5):
        super().__init__()
        self.context_enabled = context
        self.bound = bound
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 5, padding=2), nn.SiLU(), nn.Conv2d(12, 16, 3, padding=1), nn.SiLU()
        )
        self.context = nn.Sequential(
            nn.Conv1d(16, 16, 9, padding=4),
            nn.SiLU(),
            nn.Conv1d(16, 16, 9, padding=16, dilation=4),
            nn.SiLU(),
            nn.Conv1d(16, 16, 9, padding=32, dilation=8),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 12, 3, padding=1), nn.SiLU(), nn.Conv2d(12, 1, 3, padding=1)
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        if not context:
            self.context.requires_grad_(False)

    def forward(self, sdf):
        features = self.encoder(sdf.clamp(-12, 12) / 12)
        if self.context_enabled:
            sequence = F.avg_pool1d(features.mean(2), 4)
            sequence = self.context(sequence)
            sequence = F.interpolate(
                sequence, size=sdf.shape[-1], mode="linear", align_corners=False
            )
            features = features + sequence[:, :, None, :]
        correction = self.bound * torch.tanh(self.decoder(features))
        # Only change a narrow neighborhood; remote blank paper cannot become grey.
        correction = correction * (sdf.abs() < 8).to(sdf.dtype)
        field = sdf + correction
        ink = torch.sigmoid(field / 0.35)
        return ink, correction, field
