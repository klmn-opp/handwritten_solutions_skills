"""Experimental whole-word conditional VAE, isolated from the CPU runtime.

The decoder predicts an entire distance field; it does not copy font ink or
constrain corrections to the font boundary. Posterior reconstruction is an
oracle diagnostic, never evidence of text-only generation.
"""

import torch
import torch.nn.functional as F
from torch import nn


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(4, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.SiLU(),
    )


class WordStyle(nn.Module):
    def __init__(self, latent: int = 32):
        super().__init__()
        self.latent = latent
        self.enc = nn.ModuleList([block(1, 16), block(16, 24), block(24, 40), block(40, 64)])
        self.posterior = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 24, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(24, 40, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(40, 64, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.AvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(512, latent * 2),
        )
        self.bottleneck = block(64 + latent, 64)
        self.dec = nn.ModuleList([block(104, 40), block(64, 24), block(40, 16)])
        self.output = nn.Conv2d(16, 1, 3, padding=1)

    def encode(self, reference, target):
        if reference.shape[-2:] != (128, 512) or target.shape != reference.shape:
            raise ValueError("Frozen whole-word geometry must be 128x512")
        mu, logvar = self.posterior(torch.cat([reference, target], 1)).chunk(2, dim=1)
        return mu, logvar.clamp(-8, 4)

    def forward(self, reference, z):
        if z.shape != (reference.shape[0], self.latent):
            raise ValueError("One shared learned latent per complete word required")
        features = []
        x = reference
        for i, layer in enumerate(self.enc):
            x = layer(x if i == 0 else F.avg_pool2d(x, 2))
            features.append(x)
        x = self.bottleneck(
            torch.cat([x, z[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], 1)
        )
        for layer, skip in zip(self.dec, reversed(features[:-1]), strict=True):
            x = layer(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="nearest"), skip], 1))
        field = 8 * torch.tanh(self.output(x))
        return torch.sigmoid(field / 0.35), field


def reconstruction_loss(ink, field, target, target_sdf):
    # No loss toward the font. Match actual whole-word shapes at several scales.
    weight = 0.2 + torch.exp(-target_sdf.abs() / 3)
    sdf_loss = (F.smooth_l1_loss(field / 8, target_sdf / 8, reduction="none") * weight).mean()
    dice = 1 - (2 * (ink * target).sum((1, 2, 3)) + 1) / (
        ink.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1
    )
    coarse = sum(F.l1_loss(F.avg_pool2d(ink, k), F.avg_pool2d(target, k)) for k in (4, 16))
    return sdf_loss + 0.5 * dice.mean() + coarse
