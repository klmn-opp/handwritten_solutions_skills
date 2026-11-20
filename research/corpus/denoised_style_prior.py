"""Learn a style distribution without retrieving style centers at generation."""

import math

import torch
from torch import nn


class DenoisedStyle(nn.Module):
    architecture = "semantic-glyph-style-denoiser/1"

    def __init__(self):
        super().__init__()
        self.character = nn.Embedding(128, 8, padding_idx=0)
        self.network = nn.Sequential(
            nn.Linear(33, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 8)
        )

    def forward(self, tokens, noisy, sigma, support):
        normalizer = (1 + sigma.square()).sqrt()
        condition = self.character(tokens).flatten(1)
        features = torch.cat([condition, noisy / normalizer, sigma.log()], dim=-1)
        residual = self.network(features)
        return (noisy / normalizer.square() + (sigma / normalizer) * residual) * support

    @torch.inference_mode()
    def generate(self, tokens, noise, support):
        values = torch.logspace(math.log10(4), math.log10(0.03), 24, device=noise.device)
        sample = noise * support * math.sqrt(17)
        for current, following in zip(values[:-1], values[1:], strict=True):
            estimate = self(tokens, sample, current.expand(len(tokens), 1), support)
            slope = (sample - estimate) / current
            step = following - current
            extrapolate = sample + step * slope
            next_estimate = self(tokens, extrapolate, following.expand(len(tokens), 1), support)
            sample += step / 2 * (slope + (extrapolate - next_estimate) / following)
        return self(tokens, sample, values[-1].expand(len(tokens), 1), support)
