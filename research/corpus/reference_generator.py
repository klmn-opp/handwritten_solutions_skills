"""Optional Torch model: learned reference-conditioned offline ink, never stroke recovery."""

import torch
import torch.nn.functional as F
from torch import nn


def block(inputs: int, outputs: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride, 1),
        nn.GroupNorm(4, outputs),
        nn.SiLU(),
        nn.Conv2d(outputs, outputs, 3, 1, 1),
        nn.GroupNorm(4, outputs),
        nn.SiLU(),
    )


class ReferenceCVAE(nn.Module):
    """Spatial personal outline condition + learned conditional prior + SDF decoder.

    No pixel copy/addition or coordinate warp in the output path. Reference features
    are learned; the output is entirely predicted by convolutional parameters.
    One writer only. Context/online trajectory conditioning is not implemented.
    """

    def __init__(self, classes: int, latent: int = 8):
        super().__init__()
        self.spec = {
            "name": "reference-sdf-cvae/1",
            "classes": classes,
            "latent": latent,
            "size": 96,
            "writer_count": 1,
        }
        self.latent = latent
        self.character = nn.Embedding(classes, 8)
        self.e1, self.e2, self.e3 = block(1, 16, 2), block(16, 32, 2), block(32, 64, 2)
        self.posterior_encoder = nn.Sequential(
            block(2, 16, 2), block(16, 32, 2), block(32, 64, 2), nn.Flatten()
        )
        self.posterior = nn.Linear(64 * 12 * 12 + 8, latent * 2)
        self.prior = nn.Linear(64 * 12 * 12 + 8, latent * 2)
        self.d3 = block(64 + latent + 8, 64)
        self.d2 = block(64 + 32 + latent, 32)
        self.d1 = block(32 + 16 + latent, 16)
        self.output = nn.Sequential(block(16 + latent, 16), nn.Conv2d(16, 1, 1), nn.Tanh())

    def encode_reference(self, reference, labels):
        a = self.e1(reference)
        b = self.e2(a)
        c = self.e3(b)
        emb = self.character(labels)
        mu, lv = self.prior(torch.cat([c.flatten(1), emb], 1)).chunk(2, 1)
        return (a, b, c, emb), mu, lv.clamp(-8, 4)

    @staticmethod
    def spatial(value, tensor):
        return value[:, :, None, None].expand(-1, -1, tensor.shape[2], tensor.shape[3])

    def decode(self, features, z):
        a, b, c, emb = features
        x = self.d3(torch.cat([c, self.spatial(z, c), self.spatial(emb, c)], 1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.d2(torch.cat([x, b, self.spatial(z, b)], 1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.d1(torch.cat([x, a, self.spatial(z, a)], 1))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.output(torch.cat([x, self.spatial(z, x)], 1))

    def forward(self, reference, target, labels):
        features, pm, pl = self.encode_reference(reference, labels)
        encoded = self.posterior_encoder(torch.cat([reference, target], 1))
        qm, ql = self.posterior(torch.cat([encoded, features[-1]], 1)).chunk(2, 1)
        ql = ql.clamp(-8, 4)
        z = qm + (0.5 * ql).exp() * torch.randn_like(qm)
        return self.decode(features, z), (qm, ql, pm, pl)

    def generate(self, reference, labels, noise):
        features, mu, lv = self.encode_reference(reference, labels)
        return self.decode(features, mu + (0.5 * lv).exp() * noise)


def ink_from_sdf(sdf):
    # Fixed representation, shared at train and inference; not post-hoc darkening.
    return torch.sigmoid(sdf * (8 / 0.35))


def objective(prediction, target_sdf, target_ink, distribution, beta: float):
    ink = ink_from_sdf(prediction)
    boundary = 1 + 4 * (target_sdf.abs() < 0.35).float()
    sdf_loss = ((prediction - target_sdf).abs() * boundary).mean()
    bce = (F.binary_cross_entropy(ink, target_ink, reduction="none") * (1 + 5 * target_ink)).mean()
    dice = 1 - (2 * (ink * target_ink).sum((1, 2, 3)) + 1) / (
        ink.sum((1, 2, 3)) + target_ink.sum((1, 2, 3)) + 1
    )
    qm, ql, pm, pl = distribution
    kl = 0.5 * (pl - ql + (ql.exp() + (qm - pm).square()) / pl.exp() - 1).mean()
    loss = sdf_loss + 0.3 * bce + 0.2 * dice.mean() + beta * kl
    return loss, {"sdf": sdf_loss, "bce": bce, "dice": dice.mean(), "kl": kl}
