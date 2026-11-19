"""Learned coordinate-to-ink fields, conditioned on characters and their neighbours.

No reference image/outline, word-ID lookup, stored glyph bitmap or output-copy path.
This deterministic local component is NOT an end-to-end continuous-writing model.
"""

import math

import torch
from torch import nn


class SemanticInk(nn.Module):
    architecture = "semantic-neighbour-ink-field/1"

    def __init__(self, context_scale=1.0, smooth_coordinates=False):
        super().__init__()
        self.character = nn.Embedding(128, 32, padding_idx=0)
        self.neighbour = nn.Embedding(128, 12, padding_idx=0)
        self.writer = nn.Parameter(torch.zeros(16))
        self.context_scale = float(context_scale)
        frequencies = [1, 2, 0, 0] if smooth_coordinates else [1, 2, 4, 8]
        self.register_buffer("frequencies", torch.tensor(frequencies) * math.pi)
        self.field = nn.Sequential(
            nn.Linear(32 + 24 + 16 + 18, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, tokens: torch.Tensor, boxes: torch.Tensor, points: torch.Tensor):
        """B×L×3 token IDs, B×L×4 boxes, B×P×2 pixel centres → logits/SDF.

        Layout bounds are metrics, not contours. Two nearby glyph fields permit
        overlapping strokes. No real target geometry is inferred at generation.
        """
        if (
            tokens.ndim != 3
            or tokens.shape[-1] != 3
            or boxes.shape != (*tokens.shape[:2], 4)
            or points.ndim != 3
            or points.shape[0] != tokens.shape[0]
            or points.shape[-1] != 2
        ):
            raise ValueError("Batched semantic tokens, boxes and query points required")
        centers = (boxes[..., :2] + boxes[..., 2:]) / 2
        radii = ((boxes[..., 2:] - boxes[..., :2]) / 2).clamp_min(0.5)
        uv = (points[:, :, None] - centers[:, None]) / radii[:, None]
        # Nearest two ink bounding rectangles. Tie breaks are deterministic.
        score = torch.relu(uv.abs() - 1).square().sum(-1) + 0.001 * uv.square().sum(-1)
        score = score.masked_fill(tokens[:, None, :, 0] == 0, 1e12)
        ids = score.topk(min(2, tokens.shape[1]), largest=False).indices
        xy = uv.gather(2, ids[..., None].expand(-1, -1, -1, 2))
        batch = torch.arange(len(tokens), device=tokens.device)[:, None, None]
        local = tokens[batch, ids]
        angle = xy[..., None] * self.frequencies
        pe = torch.cat([xy, angle.sin().flatten(-2), angle.cos().flatten(-2)], -1)
        features = torch.cat(
            [
                self.character(local[..., 0]),
                self.neighbour(local[..., 1]) * self.context_scale,
                self.neighbour(local[..., 2]) * self.context_scale,
                self.writer.expand(*xy.shape[:-1], -1),
                pe,
            ],
            -1,
        )
        result = self.field(features)
        support = (xy.abs() <= 1.6).all(-1) & (local[..., 0] != 0)
        logits = result[..., 0].masked_fill(~support, -15).amax(-1)
        sdf = result[..., 1].tanh().masked_fill(~support, -1).amax(-1)
        return logits, sdf

    @torch.inference_mode()
    def image(self, tokens, boxes, height=128, width=512, chunk=8192):
        y, x = torch.meshgrid(
            torch.arange(height, device=boxes.device),
            torch.arange(width, device=boxes.device),
            indexing="ij",
        )
        points = torch.stack([x.flatten(), y.flatten()], -1).float()
        pieces = []
        for part in points.split(chunk):
            p = part[None].expand(len(tokens), -1, -1)
            pieces.append(self(tokens, boxes, p)[0].sigmoid())
        return torch.cat(pieces, 1).reshape(len(tokens), height, width)
