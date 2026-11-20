"""Frozen local neural glyph composition, explicitly NOT contextual generation."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def compose(model, tokens, boxes, style, height, width, *, tile_width=512, return_layers=False):
    if tokens.ndim != 2 or tokens.shape[-1] != 3:
        raise ValueError("One semantic token sequence required")
    if boxes.shape != (len(tokens), 4) or style.shape != (len(tokens), 8):
        raise ValueError("One layout box and condition per token required")
    if min(height, width, tile_width) <= 0:
        raise ValueError("Positive canvas and tile sizes required")
    valid = tokens[:, 0] != 0
    if not bool(valid.any()):
        raise ValueError("Nonempty content required")
    boxes, style, labels = boxes[valid], style[valid], tokens[valid, 0]
    if not bool(torch.isfinite(boxes).all()) or not bool((boxes[:, 2:] > boxes[:, :2]).all()):
        raise ValueError("Finite positive layout boxes required")
    ink = model.image(labels, style)[:, None]
    if ink.shape[-2:] != (96, 96):
        raise ValueError("Canonical 96px decoder required")
    pieces = []
    for start in range(0, width, tile_width):
        end = min(width, start + tile_width)
        y, x = torch.meshgrid(
            torch.arange(height, device=boxes.device),
            torch.arange(start, end, device=boxes.device),
            indexing="ij",
        )
        positions = torch.stack([x, y], -1).float()
        # The supervised ink boxes are [16,16,80,80] within canonical96 images.
        local = 16 + 64 * (positions[None] - boxes[:, None, None, :2]) / (
            boxes[:, None, None, 2:] - boxes[:, None, None, :2]
        )
        sampled = F.grid_sample(
            ink, local * (2 / 95) - 1, align_corners=True, padding_mode="zeros", mode="bilinear"
        )[:, 0]
        pieces.append(sampled if return_layers else sampled.amax(0))
    return torch.cat(pieces, dim=-1)
