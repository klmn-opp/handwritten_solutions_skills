"""Learned word-shared/local latent prior for a frozen neural ink decoder."""

import torch
from torch import nn


class HierarchicalInkStyle(nn.Module):
    architecture = "hierarchical-neural-ink-style/1"

    def __init__(self, *, context=True):
        super().__init__()
        self.context = context
        self.character = nn.Embedding(128, 8, padding_idx=0)
        self.writer = nn.Parameter(torch.zeros(8))
        self.encode = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.word_posterior = nn.Linear(32, 16)
        self.local_posterior = nn.Linear(40, 8)
        self.decode = nn.Sequential(
            nn.Linear(44, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 8)
        )

    def posterior(self, tokens, codes, known):
        if tokens.ndim != 3 or tokens.shape[-1] != 3 or codes.shape != (*tokens.shape[:2], 8):
            raise ValueError("Batched semantic sequences and numerical styles required")
        if known.shape != tokens.shape[:2] or not bool(known.any(-1).all()):
            raise ValueError("Every supervised word needs at least one verified local observation")
        features = self.encode(torch.cat([self.character(tokens[..., 0]), codes], -1))
        pooled = (features * known[..., None]).sum(1) / known.sum(1, keepdim=True)
        wm, wv = self.word_posterior(pooled).chunk(2, -1)
        lm, lv = self.local_posterior(
            torch.cat([features, wm[:, None].expand(-1, tokens.shape[1], -1)], -1)
        ).chunk(2, -1)
        return wm, wv.clamp(-6, 3), lm, lv.clamp(-6, 3)

    def forward(self, tokens, word_latent, local_latent):
        current = self.character(tokens[..., 0])
        if self.context:
            previous, following = self.character(tokens[..., 1]), self.character(tokens[..., 2])
        else:
            previous, following = torch.zeros_like(current), torch.zeros_like(current)
        writer = self.writer.expand(*tokens.shape[:2], -1)
        shared = word_latent[:, None].expand(-1, tokens.shape[1], -1)
        result = self.decode(
            torch.cat([current, previous, following, writer, shared, local_latent], -1)
        )
        return 3.5 * (result / 3.5).tanh() * (tokens[..., 0] != 0)[..., None]

    @staticmethod
    def kl(mean, logvar):
        return 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean(-1)
