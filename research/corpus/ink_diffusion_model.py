"""Small conditional ink denoiser, never imported by the CPU runtime.

Unlike the geometric baseline, every output pixel is predicted. Personal font
ink is an input condition, not a copied output, residual anchor or loss target.
The Gaussian initial state supplies stochasticity without a learned posterior.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.first = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, cout)
        self.condition = nn.Linear(128, 2 * cout)
        self.second = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, condition):
        h = self.norm1(self.first(x))
        scale, bias = self.condition(condition).chunk(2, dim=1)
        h = F.silu(h * (1 + scale[:, :, None, None]) + bias[:, :, None, None])
        h = self.norm2(self.second(h))
        return F.silu(h + self.skip(x))


class InkDenoiser(nn.Module):
    architecture = "real-ink-conditional-v-denoiser/1"

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(129, 8, padding_idx=0)
        self.context = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 64))
        self.time = nn.Sequential(nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 64))
        self.enc = nn.ModuleList(
            [Residual(2, 12), Residual(12, 24), Residual(24, 48), Residual(48, 64)]
        )
        self.mid = Residual(64, 64)
        self.dec = nn.ModuleList([Residual(112, 48), Residual(72, 24), Residual(36, 12)])
        self.output = nn.Conv2d(12, 1, 3, padding=1)

    def forward(self, noisy, reference, tokens, time):
        if noisy.shape != reference.shape or noisy.shape[1:] != (1, 128, 512):
            raise ValueError("Complete 128x512 image conditions required")
        if tokens.shape != (len(noisy), 32) or time.shape != (len(noisy),):
            raise ValueError("Exact token and diffusion-time batch required")
        frequencies = torch.exp(torch.arange(16, device=time.device) * (-math.log(10000) / 15))
        phase = time[:, None] * frequencies[None] * 1000
        c = torch.cat(
            [
                self.time(torch.cat([phase.sin(), phase.cos()], 1)),
                self.context(self.embedding(tokens).flatten(1)),
            ],
            1,
        )
        h = torch.cat([noisy, reference], 1)
        features = []
        for i, layer in enumerate(self.enc):
            h = layer(h if i == 0 else F.avg_pool2d(h, 2), c)
            features.append(h)
        h = self.mid(h, c)
        for layer, skip in zip(self.dec, reversed(features[:-1]), strict=True):
            h = layer(
                torch.cat([F.interpolate(h, size=skip.shape[-2:], mode="nearest"), skip], 1), c
            )
        return self.output(h)


def schedule(t):
    angle = t * (math.pi / 2)
    return angle.cos()[:, None, None, None], angle.sin()[:, None, None, None]


def diffusion_loss(model, field, reference, tokens, t, noise):
    alpha, sigma = schedule(t)
    noisy = alpha * field + sigma * noise
    velocity = alpha * noise - sigma * field
    predicted = model(noisy, reference, tokens, t)
    # No target toward the reference; no OCR teacher, morphology or field warp.
    return F.mse_loss(predicted, velocity)


@torch.inference_mode()
def generate(model, reference, tokens, noise, steps=64):
    if steps < 2 or noise.shape != reference.shape:
        raise ValueError("Independent full-size noise and at least two reverse steps required")
    x = noise.clone()
    times = torch.linspace(1, 0, steps + 1, device=x.device)
    for i in range(steps):
        t = times[i].expand(len(x))
        alpha, sigma = schedule(t)
        v = model(x, reference, tokens, t)
        field = (alpha * x - sigma * v).clamp(-1, 1)
        epsilon = sigma * x + alpha * v
        next_alpha, next_sigma = schedule(times[i + 1].expand(len(x)))
        x = next_alpha * field + next_sigma * epsilon
    return torch.sigmoid(x * (8 / 0.35)), x
