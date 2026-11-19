"""Optional small whole-region generator; no character slicing or pixel-copy output."""

import torch
import torch.nn.functional as F
from torch import nn

from research.corpus.reference_generator import block


class ContextInk(nn.Module):
    """Convolutional content reference, line context, and one shared region latent.

    Learns offline words/short lines as complete images. One personal writer, no
    multi-writer style transfer claim; mathematical structure is outside this model.
    """

    def __init__(self, context: bool = True, global_style: bool = True):
        super().__init__()
        self.spec = {
            "name": "context-ink-region/1",
            "height": 64,
            "width": 512,
            "latent": 8,
            "context": context,
            "global_style": global_style,
            "writer_count": 1,
            "math_generation": False,
        }
        self.e1, self.e2, self.e3 = block(1, 12, 2), block(12, 24, 2), block(24, 48, 2)
        self.line_context = nn.Sequential(
            nn.Conv1d(48, 48, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(48, 48, 5, padding=4, dilation=2),
            nn.SiLU(),
            nn.Conv1d(48, 48, 5, padding=8, dilation=4),
        )
        self.posterior_encoder = nn.Sequential(
            block(2, 12, 2),
            block(12, 24, 2),
            block(24, 48, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.posterior = nn.Linear(48, 16)
        self.prior = nn.Linear(48, 16)
        self.role = nn.Embedding(2, 4)
        self.d3 = block(48 + 8 + 4, 48)
        self.d2 = block(48 + 24 + 8, 24)
        self.d1 = block(24 + 12 + 8, 12)
        self.output = nn.Sequential(
            nn.Conv2d(12 + 8, 12, 3, padding=1), nn.SiLU(), nn.Conv2d(12, 1, 1), nn.Tanh()
        )

    @staticmethod
    def expand(value, image):
        return value[:, :, None, None].expand(-1, -1, *image.shape[-2:])

    def encode(self, reference):
        a = self.e1(reference)
        b = self.e2(a)
        c = self.e3(b)
        if self.spec["context"]:
            c = c + self.line_context(c.mean(2))[:, :, None, :]
        pm, pl = self.prior(c.mean((2, 3))).chunk(2, 1)
        return (a, b, c), pm, pl.clamp(-6, 2)

    def decode(self, features, z, role):
        if not self.spec["global_style"]:
            z = torch.zeros_like(z)
        a, b, c = features
        x = self.d3(torch.cat([c, self.expand(z, c), self.expand(self.role(role), c)], 1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.d2(torch.cat([x, b, self.expand(z, b)], 1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.d1(torch.cat([x, a, self.expand(z, a)], 1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.output(torch.cat([x, self.expand(z, x)], 1))

    def forward(self, reference, target, role, noise):
        features, pm, pl = self.encode(reference)
        qm, ql = self.posterior(self.posterior_encoder(torch.cat([reference, target], 1))).chunk(
            2, 1
        )
        ql = ql.clamp(-6, 2)
        z = qm + (ql * 0.5).exp() * noise
        return self.decode(features, z, role), (qm, ql, pm, pl)

    def generate(self, reference, role, noise):
        features, mu, lv = self.encode(reference)
        return self.decode(features, mu + (lv * 0.5).exp() * noise, role)
