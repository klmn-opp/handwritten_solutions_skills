"""Experimental canonical image decoder; no bitmap or outline input.

This local component alone is not a contextual word/line generator.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CanonicalInk(nn.Module):
    architecture = "canonical-conditioned-ink/1"

    def __init__(self):
        super().__init__()
        self.character = nn.Embedding(128, 16, padding_idx=0)
        self.writer = nn.Parameter(torch.zeros(8))
        self.expand = nn.Sequential(nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 32 * 6 * 6))
        self.blocks = nn.ModuleList()
        self.modulation = nn.ModuleList()
        previous = 32
        for channels in [32, 24, 16, 8]:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous, channels, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.SiLU(),
                )
            )
            modulation = nn.Linear(8, 2 * channels)
            nn.init.zeros_(modulation.weight)
            nn.init.zeros_(modulation.bias)
            self.modulation.append(modulation)
            previous = channels
        self.output = nn.Conv2d(8, 1, 1)
        nn.init.constant_(self.output.bias, -2)

    def forward(self, characters, local):
        if characters.ndim != 1 or local.shape != (len(characters), 8):
            raise ValueError("One semantic identity and eight numerical conditions per sample")
        if not bool(((characters > 0) & (characters < 128)).all()):
            raise ValueError("Nonempty supported semantic identity required")
        x = torch.cat([self.character(characters), local, self.writer.expand(len(local), -1)], -1)
        x = self.expand(x).reshape(-1, 32, 6, 6)
        for block, modulation in zip(self.blocks, self.modulation, strict=True):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            scale, shift = modulation(local).chunk(2, dim=-1)
            x = block[0](x)
            x = block[1](
                x * (1 + 0.5 * scale[:, :, None, None].tanh())
                + 0.5 * shift[:, :, None, None].tanh()
            )
            x = block[3](block[2](x))
        return self.output(x)[:, 0]

    @torch.inference_mode()
    def image(self, characters, local):
        return self(characters, local).sigmoid()
