"""Pure semantic/layout conditions -> jointly predicted offline ink pixels.

No input image, reference outline, bitmap lookup, or output-copy path. Spatial
interaction is an architectural capability, not proof of learned natural joins.
"""

import math

import torch
from torch import nn


class SemanticCanvasInk(nn.Module):
    architecture = "semantic-canvas-ink/1"
    receptive_field = 63
    halo = 31

    def __init__(self):
        super().__init__()
        self.character = nn.Embedding(128, 16, padding_idx=0)
        self.local = nn.Linear(8, 8, bias=False)
        nn.init.zeros_(self.local.weight)
        self.writer = nn.Parameter(torch.zeros(8))
        self.register_buffer("frequencies", torch.tensor([1, 2, 4, 8]) * math.pi)
        # Two local fields, each: identity16 + morphology8 + position18 + scale2.
        self.stem = nn.Sequential(nn.Conv2d(96, 24, 3, padding=1), nn.SiLU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(24, 24, 3, dilation=d, padding=d),
                    nn.SiLU(),
                    nn.Conv2d(24, 24, 3, dilation=d, padding=d),
                    nn.SiLU(),
                )
                for d in [1, 2, 4, 8]
            ]
        )
        self.output = nn.Conv2d(24, 1, 1)
        nn.init.constant_(self.output.bias, -2)

    def conditions(self, tokens, boxes, height, width, style=None, x_offset=0):
        if tokens.ndim != 3 or tokens.shape[-1] != 3:
            raise ValueError("Batched character and neighbour IDs required")
        if boxes.shape != (*tokens.shape[:2], 4) or height <= 0 or width <= 0:
            raise ValueError("Matching layout bounds and positive image size required")
        if not bool((tokens[..., 0] != 0).any(1).all()):
            raise ValueError("Each sample requires nonempty semantic content")
        if style is None:
            style = boxes.new_zeros(*tokens.shape[:2], 8)
        if style.shape != (*tokens.shape[:2], 8):
            raise ValueError("Eight numerical local conditions per token required")
        if tokens.shape[1] == 1:
            tokens = torch.cat([tokens, torch.zeros_like(tokens)], 1)
            boxes = torch.cat([boxes, torch.zeros_like(boxes)], 1)
            style = torch.cat([style, torch.zeros_like(style)], 1)
        centers = (boxes[..., :2] + boxes[..., 2:]) / 2
        radii = ((boxes[..., 2:] - boxes[..., :2]) / 2).clamp_min(0.5)
        y, x = torch.meshgrid(
            torch.arange(height, device=boxes.device),
            torch.arange(width, device=boxes.device) + x_offset,
            indexing="ij",
        )
        points = torch.stack([x.flatten(), y.flatten()], -1).float()
        uv = (points[None, :, None] - centers[:, None]) / radii[:, None]
        score = torch.relu(uv.abs() - 1).square().sum(-1) + 0.001 * uv.square().sum(-1)
        score = score.masked_fill(tokens[:, None, :, 0] == 0, 1e12)
        ids = score.topk(2, largest=False).indices
        batch = torch.arange(len(tokens), device=tokens.device)[:, None, None]
        xy = uv.gather(2, ids[..., None].expand(-1, -1, -1, 2))
        chars = tokens[batch, ids, 0]
        supported = (xy.abs() <= 1.6).all(-1) & (chars != 0)
        xy = xy.clamp(-2, 2)
        angle = xy[..., None] * self.frequencies
        position = torch.cat([xy, angle.sin().flatten(-2), angle.cos().flatten(-2)], -1)
        features = torch.cat(
            [
                self.character(chars),
                self.local(style[batch, ids]),
                position,
                radii[batch, ids].log() / 4,
            ],
            -1,
        )
        features = features.masked_fill(~supported[..., None], 0).flatten(-2)
        writer = self.writer.expand(*features.shape[:-1], -1)
        return (
            torch.cat([features, writer], -1)
            .transpose(1, 2)
            .reshape(len(tokens), 96, height, width)
        )

    def forward(self, tokens, boxes, height, width, *, style=None, x_offset=0):
        h = self.stem(self.conditions(tokens, boxes, height, width, style, x_offset))
        for block in self.blocks:
            h = h + 0.2 * block(h)
        return self.output(h)[:, 0]

    @torch.inference_mode()
    def image(self, tokens, boxes, height=128, width=512, *, style=None, tile_width=512):
        if tile_width <= 0:
            raise ValueError("Positive tile width required")
        pieces = []
        for start in range(0, width, tile_width):
            end = min(width, start + tile_width)
            left, right = max(0, start - self.halo), min(width, end + self.halo)
            part = self(tokens, boxes, height, right - left, style=style, x_offset=left)
            pieces.append(part[:, :, start - left : end - left].sigmoid())
        return torch.cat(pieces, -1)
