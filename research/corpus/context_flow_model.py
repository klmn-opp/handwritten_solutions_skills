"""Learn a smooth contextual deformation of personal contours, not random jitter.

The prior remains explicit: this model cannot invent arbitrary missing letters or
new stroke topology. The sampler differentiates coordinates, never font pixels.
"""

import torch
import torch.nn.functional as F
from torch import nn

from research.corpus.context_contour_model import ContourInk


def sample_fixed_field(field, flow):
    """Bilinear pixel-coordinate sampler with deterministic coordinate gradients."""
    n, _, h, w = field.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=field.device), torch.arange(w, device=field.device), indexing="ij"
    )
    x = (xx[None] + flow[:, 0]).clamp(0, w - 1)
    y = (yy[None] + flow[:, 1]).clamp(0, h - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=w - 1), (y0 + 1).clamp(max=h - 1)
    wx, wy = x - x0, y - y0
    flat = field.detach().reshape(n, h * w)

    def take(a, b):
        return torch.gather(flat, 1, (b * w + a).reshape(n, -1)).reshape(n, h, w)

    value = (
        take(x0, y0) * (1 - wx) * (1 - wy)
        + take(x1, y0) * wx * (1 - wy)
        + take(x0, y1) * (1 - wx) * wy
        + take(x1, y1) * wx * wy
    )
    return value[:, None]


def jacobian_determinant(flow):
    dx = flow[..., :-1, 1:] - flow[..., :-1, :-1]
    dy = flow[..., 1:, :-1] - flow[..., :-1, :-1]
    return (1 + dx[:, 0]) * (1 + dy[:, 1]) - dx[:, 1] * dy[:, 0]


class FlowInk(ContourInk):
    def __init__(self):
        super().__init__(context=True, bound=1.0)
        self.decoder[-1] = nn.Conv2d(12, 3, 3, padding=1)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.last_flow = None

    def forward(self, sdf):
        raw = self.decoder(self.features(sdf))
        smooth = F.avg_pool2d(torch.tanh(raw[:, :2]), 17, stride=1, padding=8)
        flow = smooth * torch.tensor([8.0, 5.0], device=sdf.device)[None, :, None, None]
        field = sample_fixed_field(sdf, flow)
        correction = torch.tanh(raw[:, 2:3]) * (sdf.abs() < 8).to(sdf.dtype)
        field = field + correction
        self.last_flow = flow
        return torch.sigmoid(field / 0.35), correction, field
