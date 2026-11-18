"""Optional small conditional VAE research model. Import only in the training environment.

Offline raster generation, not online trajectory recovery. No model is published by this module.
"""

import torch
from torch import nn
from torch.nn import functional as F


class GlyphCVAE(nn.Module):
    def __init__(self, classes: int, writers: int = 1, latent: int = 16):
        super().__init__()
        self.spec = {
            "architecture": "glyph-cvae/1",
            "classes": classes,
            "writers": writers,
            "latent": latent,
            "size": 64,
            "output": "offline-raster-ink",
        }
        self.character = nn.Embedding(classes, 16)
        self.writer = nn.Embedding(writers, 8)
        # Context is [relative scale, preceding gap, following gap, baseline offset].
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.posterior = nn.Linear(64 * 8 * 8 + 28, latent * 2)
        self.project = nn.Linear(latent + 28, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
        )

    def condition(self, label, writer, context):
        return torch.cat([self.character(label), self.writer(writer), context], dim=1)

    def generate(self, label, writer, context, z):
        features = self.project(torch.cat([z, self.condition(label, writer, context)], dim=1))
        return self.decoder(features.reshape(-1, 64, 8, 8)).sigmoid()

    def forward(self, ink, label, writer, context):
        mu, logvar = self.posterior(
            torch.cat([self.encoder(ink), self.condition(label, writer, context)], dim=1)
        ).chunk(2, dim=1)
        logvar = logvar.clamp(-12, 8)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        generated = self.generate(label, writer, context, z)
        return generated, mu, logvar


def objective(prediction, target, mu, logvar, beta=0.001):
    # Weighted reconstruction prevents a blank-page solution from dominating sparse ink.
    reconstruction = (
        (1 + target * 8) * F.binary_cross_entropy(prediction, target, reduction="none")
    ).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    return reconstruction + beta * kl, {
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
    }
