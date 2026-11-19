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


def monotone_controls(raw):
    """Construct a positive-Jacobian triangular map; no learned fold penalty needed."""
    if raw.shape[1:] != (3, 17):
        raise ValueError("Expected horizontal increments, vertical scale and shift")
    intervals = 0.25 + 0.75 * (16 * torch.softmax(raw[:, 0, :16], dim=1))
    nodes = torch.cat([torch.zeros_like(intervals[:, :1]), intervals.cumsum(1)], dim=1)
    x = torch.linspace(0, 512 - 1, 17, device=raw.device, dtype=raw.dtype)
    dx = nodes * ((512 - 1) / 16) - x
    # Uniform convex blend with identity retains strict monotonicity.
    dx = dx * (64 / dx.abs().amax(1, keepdim=True).clamp(min=64))
    yy = torch.linspace(0, 128 - 1, 5, device=raw.device, dtype=raw.dtype) - 63.5
    dy = (
        0.2 * torch.tanh(raw[:, 1])[:, None] * yy[None, :, None]
        + 8 * torch.tanh(raw[:, 2])[:, None]
    )
    return torch.stack([dx[:, None].expand(-1, 5, -1), dy], dim=1)


def render(reference, controls):
    """Warp continuous source coverage without thresholding or negative ink correction."""
    return sample_fixed_field(reference, dense_flow(controls, reference.shape[-2:]))


class AlignedStyle(nn.Module):
    """Content-conditioned distribution over registered real-word geometry.

    Prior inference uses only font content, text tokens and random latent. The
    target field is confined to the posterior training/oracle path.
    """

    def __init__(self, stochastic=True, latent=4, monotone=False):
        super().__init__()
        self.stochastic, self.latent, self.monotone = stochastic, latent, monotone
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
            nn.Linear(128 + latent, 192),
            nn.SiLU(),
            nn.Linear(192, 51 if monotone else 170),
            nn.Tanh(),
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
        raw = self.decoder(torch.cat([condition, z], dim=1))
        return (
            monotone_controls(raw.reshape(-1, 3, 17))
            if self.monotone
            else raw.reshape(-1, 2, *GRID) * self.bounds
        )

    def generate(self, reference, tokens, epsilon):
        condition = self.condition(reference, tokens)
        mu, logvar = self.distribution(condition)
        z = mu + (0.5 * logvar).exp() * epsilon if self.stochastic else torch.zeros_like(mu)
        controls = self.decode(condition, z)
        return render(reference, controls), controls
