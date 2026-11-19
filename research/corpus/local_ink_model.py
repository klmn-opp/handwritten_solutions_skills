"""Deterministic local ink prediction from content outlines; no image copy output path."""

import torch
from torch import nn


class LocalInk(nn.Module):
    architecture = "canonical-context-local-ink/1"
    receptive_field = 63

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.SiLU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(16, 16, 3, padding=d, dilation=d),
                    nn.SiLU(),
                    nn.Conv2d(16, 16, 3, padding=d, dilation=d),
                    nn.SiLU(),
                )
                for d in [1, 2, 4, 8]
            ]
        )
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 4 or reference.shape[1] != 1:
            raise ValueError("Batched one-channel content ink required")
        h = self.stem(reference)
        for block in self.blocks:
            h = h + 0.2 * block(h)
        return self.output(h)
