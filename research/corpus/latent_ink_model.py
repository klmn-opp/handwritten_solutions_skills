"""Small learned offline ink codec and conditional latent generator; GPU research only."""

import torch
import torch.nn.functional as F
from torch import nn

from research.corpus.ink_diffusion_model import Residual, schedule


class InkCodec(nn.Module):
    architecture = "personal-spatial-ink-codec/1"

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(16, 24, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(24, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 8, 3, padding=1),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 24, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(24, 16, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def encode(self, ink):
        if ink.shape[1:] != (1, 128, 512):
            raise ValueError("Complete 128x512 ink required")
        return self.encoder(ink)

    def decode(self, latent):
        if latent.shape[1:] != (8, 16, 64):
            raise ValueError("Complete spatial ink code required")
        return self.decoder(latent)

    def forward(self, ink):
        return self.decode(self.encode(ink))


def reconstruction_loss(logits, target):
    weights = 1 + 7 * target
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum((1, 2, 3)) + 1) / (
        probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1
    )
    return bce + 0.5 * dice.mean()


class LatentInk(nn.Module):
    architecture = "text-conditioned-spatial-ink-denoiser/1"

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(129, 8, padding_idx=0)
        self.context = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 64))
        self.time = nn.Sequential(nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 64))
        self.enc = nn.ModuleList([Residual(16, 32), Residual(32, 48), Residual(48, 64)])
        self.mid = Residual(64, 64)
        self.dec = nn.ModuleList([Residual(112, 48), Residual(80, 32)])
        self.out = nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, noisy, reference, tokens, time):
        if noisy.shape != reference.shape or noisy.shape[1:] != (8, 16, 64):
            raise ValueError("Matched spatial codes required")
        if tokens.shape != (len(noisy), 32) or time.shape != (len(noisy),):
            raise ValueError("Exact text/time batch required")
        frequencies = torch.exp(torch.arange(16, device=time.device) * (-9.210340371976184 / 15))
        phase = time[:, None] * frequencies[None] * 1000
        condition = torch.cat(
            [
                self.time(torch.cat([phase.sin(), phase.cos()], 1)),
                self.context(self.embedding(tokens).flatten(1)),
            ],
            1,
        )
        h = torch.cat([noisy, reference], 1)
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


def bounded_code(code, mean, std):
    return torch.maximum(torch.minimum(code, (1 - mean) / std), (-1 - mean) / std)


@torch.inference_mode()
def generate(model, codec, reference, tokens, noise, mean, std, steps=64):
    if steps < 2 or noise.shape != reference.shape:
        raise ValueError("Independent matching noise and inverse schedule required")
    x = noise.clone()
    times = torch.linspace(1, 0, steps + 1, device=x.device)
    for i in range(steps):
        a, s = schedule(times[i].expand(len(x)))
        v = model(x, reference, tokens, times[i].expand(len(x)))
        clean = bounded_code(a * x - s * v, mean, std)
        eps = s * x + a * v
        na, ns = schedule(times[i + 1].expand(len(x)))
        x = na * clean + ns * eps
    return codec.decode(x * std + mean).sigmoid(), x
