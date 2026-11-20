"""Bounded learned spatial correction of internally neural-generated word ink.

A context adapter on a frozen local generator, not an end-to-end text-line model.
"""

import torch
from torch import nn


class WordContextInk(nn.Module):
    architecture = "bounded-word-ink-context/1"
    halo = 4

    def __init__(self):
        super().__init__()
        self.field = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)

    def forward(self, base, *, return_delta=False):
        if base.ndim != 4 or base.shape[1] != 1:
            raise ValueError("Batched one-channel internally generated ink required")
        delta = 3 * self.field(base).tanh()
        e = delta.exp() - 1
        # Exact identity when delta=0; exact zero/one endpoints; bounded odds change.
        result = base + base * (1 - base) * e / (1 + base * e)
        return (result, delta) if return_delta else result

    @torch.inference_mode()
    def image(self, base, *, tile_width=512):
        if tile_width <= 0:
            raise ValueError("Positive tile size required")
        parts = []
        width = base.shape[-1]
        for start in range(0, width, tile_width):
            end = min(width, start + tile_width)
            left, right = max(0, start - self.halo), min(width, end + self.halo)
            part = self(base[..., left:right])
            parts.append(part[..., start - left : end - left])
        return torch.cat(parts, dim=-1)
