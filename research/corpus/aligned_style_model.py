"""Experimental learned whole-word geometry. No claim of new ink topology."""

import torch
import torch.nn.functional as F
from torch import nn

from research.corpus.context_flow_model import sample_fixed_field

GRID = (5, 17)
BOUNDS = (64.0, 24.0)


def dense_flow(controls, size=(128, 512)):
    if controls.ndim != 4 or controls.shape[1:] != (2, *GRID):
        raise ValueError("Expected per-word 2x5x17 sampling controls")
    return F.interpolate(controls, size=size, mode="bilinear", align_corners=True)


def render(reference, controls):
    """Warp continuous source coverage without thresholding or negative ink correction."""
    return sample_fixed_field(reference, dense_flow(controls, reference.shape[-2:]))


class AlignedStyle(nn.Module):
    """Content-conditioned distribution over registered real-word geometry.

    Prior inference uses only font content, text tokens and random latent. The
    target field is confined to the posterior training/oracle path.
    """

    def __init__(self, stochastic=True, latent=4):
        super().__init__()
        self.stochastic, self.latent = stochastic, latent
        self.visual = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 24, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.AvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(768, 64),
            nn.SiLU(),
        )
        self.embedding = nn.Embedding(129, 8, padding_idx=0)
        self.text = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8, 64), nn.SiLU())
        self.prior = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, latent * 2))
        self.posterior = nn.Sequential(
            nn.Linear(128 + 170, 128), nn.SiLU(), nn.Linear(128, latent * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128 + latent, 192), nn.SiLU(), nn.Linear(192, 170), nn.Tanh()
        )
        self.register_buffer("bounds", torch.tensor(BOUNDS)[None, :, None, None])

    def condition(self, reference, tokens):
        if reference.shape[1:] != (1, 128, 512) or tokens.shape != (len(reference), 32):
            raise ValueError("Fixed whole-word geometry and exact ASCII token sequence required")
        return torch.cat([self.visual(reference), self.text(self.embedding(tokens))], dim=1)

    def distribution(self, condition, target=None):
        raw = (
            self.prior(condition)
            if target is None
            else self.posterior(torch.cat([condition, (target / self.bounds).flatten(1)], dim=1))
        )
        mu, logvar = raw.chunk(2, dim=1)
        return mu, logvar.clamp(-8, 3)

    def decode(self, condition, z):
        return self.decoder(torch.cat([condition, z], dim=1)).reshape(-1, 2, *GRID) * self.bounds

    def generate(self, reference, tokens, epsilon):
        condition = self.condition(reference, tokens)
        mu, logvar = self.distribution(condition)
        z = mu + (0.5 * logvar).exp() * epsilon if self.stochastic else torch.zeros_like(mu)
        controls = self.decode(condition, z)
        return render(reference, controls), controls
