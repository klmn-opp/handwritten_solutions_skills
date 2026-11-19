"""Shared personal parameter plus spatial text attention; no target/reference copying."""

import torch
import torch.nn.functional as F
from torch import nn

from research.corpus.latent_ink_model import LatentInk


class WriterInk(LatentInk):
    architecture = "shared-writer-text-attention-ink/1"

    def __init__(self):
        super().__init__()
        self.writer = nn.Parameter(torch.randn(32) * 0.02)
        self.writer_projection = nn.Linear(32, 128, bias=False)
        self.letter = nn.Embedding(129, 64, padding_idx=0)
        self.letter_order = nn.Embedding(32, 64)
        self.query_position = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64))
        self.text_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.attention_norm = nn.LayerNorm(64)
        self.position_strength = nn.Parameter(torch.tensor(-2.0))
        self.attention_gain = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, noisy, reference, tokens, time, positions, *, writer_enabled=True, text_enabled=True
    ):
        if noisy.shape != reference.shape or noisy.shape[1:] != (8, 16, 64):
            raise ValueError("Matching spatial ink required")
        if tokens.shape != (len(noisy), 32) or positions.shape != (*tokens.shape, 2):
            raise ValueError("Complete text and reference-only positions required")
        if (tokens != 0).sum(1).min() == 0:
            raise ValueError("Empty content condition")
        frequencies = torch.exp(torch.arange(16, device=time.device) * (-9.210340371976184 / 15))
        phase = time[:, None] * frequencies[None] * 1000
        condition = torch.cat(
            [
                self.time(torch.cat([phase.sin(), phase.cos()], 1)),
                self.context(self.embedding(tokens).flatten(1)),
            ],
            1,
        )
        if writer_enabled:
            style = F.normalize(self.writer, dim=0)
            condition = condition + self.writer_projection(style)[None]
        h = torch.cat([noisy, reference], 1)
        features = []
        for i, layer in enumerate(self.enc):
            h = layer(h if i == 0 else F.avg_pool2d(h, 2), condition)
            features.append(h)
        h = self.mid(h, condition)
        if text_enabled:
            batch, channels, height, width = h.shape
            yy, xx = torch.meshgrid(
                (torch.arange(height, device=h.device) + 0.5) / height,
                (torch.arange(width, device=h.device) + 0.5) / width,
                indexing="ij",
            )
            xy = torch.stack([xx.flatten(), yy.flatten()], 1)
            query = h.flatten(2).transpose(1, 2) + self.query_position(xy)[None]
            letters = (
                self.letter(tokens) + self.letter_order(torch.arange(32, device=h.device))[None]
            )
            center, span = positions.unbind(-1)
            bias = (
                -0.5
                * (
                    (xx.flatten()[None, :, None] - center[:, None]) / (span[:, None] * 0.8 + 0.06)
                ).square()
            )
            bias = bias * self.position_strength.sigmoid()
            bias = bias.masked_fill((tokens == 0)[:, None], float("-inf"))
            attended, _ = self.text_attention(
                query, letters, letters, attn_mask=bias.repeat_interleave(4, 0), need_weights=False
            )
            h = h + self.attention_gain.tanh() * self.attention_norm(attended).transpose(
                1, 2
            ).reshape(batch, channels, height, width)
        for layer, skip in zip(self.dec, reversed(features[:-1]), strict=True):
            h = layer(
                torch.cat([F.interpolate(h, size=skip.shape[-2:], mode="nearest"), skip], 1),
                condition,
            )
        return self.out(h)
