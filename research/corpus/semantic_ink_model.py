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

    def neighbour_features(self, ids):
        return self.neighbour(ids) * self.context_scale

    def character_features(self, ids, radii):
        return self.character(ids)

    def writer_features(self, xy, style):
        if style is not None:
            raise ValueError("This deterministic model has no random word variable")
        return self.writer.expand(*xy.shape[:-1], -1)

    def local_coordinates(self, xy, local, style):
        return xy

    def forward(
        self, tokens: torch.Tensor, boxes: torch.Tensor, points: torch.Tensor, *, style=None
    ):
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
        local_radii = radii[batch, ids]
        xy = self.local_coordinates(xy, local, style)
        angle = xy[..., None] * self.frequencies
        pe = torch.cat([xy, angle.sin().flatten(-2), angle.cos().flatten(-2)], -1)
        features = torch.cat(
            [
                self.character_features(local[..., 0], local_radii),
                self.neighbour_features(local[..., 1]),
                self.neighbour_features(local[..., 2]),
                self.writer_features(xy, style),
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
    def image(self, tokens, boxes, height=128, width=512, chunk=8192, *, style=None):
        y, x = torch.meshgrid(
            torch.arange(height, device=boxes.device),
            torch.arange(width, device=boxes.device),
            indexing="ij",
        )
        points = torch.stack([x.flatten(), y.flatten()], -1).float()
        pieces = []
        for part in points.split(chunk):
            p = part[None].expand(len(tokens), -1, -1)
            pieces.append(self(tokens, boxes, p, style=style)[0].sigmoid())
        return torch.cat(pieces, 1).reshape(len(tokens), height, width)


class ContextualSemanticInk(SemanticInk):
    """Activate learned neighbour influence continuously from a frozen identity field."""

    architecture = "semantic-neighbour-ink-field/2"

    def __init__(self):
        super().__init__(context_scale=1.0)
        self.context_gate = nn.Parameter(torch.zeros(()))

    def neighbour_features(self, ids):
        return self.neighbour(ids) * self.context_gate.tanh()

    def load_identity(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys != ["context_gate"] or result.unexpected_keys:
            raise ValueError("Only the new zero context gate may be absent")


class AutoDecodedSemanticInk(ContextualSemanticInk):
    """Training-only optimized word codes; generation takes a sampled 2-D code.

    The decoder has no sample-ID table, posterior image input or stored outlines.
    Fitting training codes alone cannot establish a useful prior or personal style.
    """

    architecture = "semantic-word-autodecoder/1"
    latent = 2

    def __init__(self):
        super().__init__()
        self.style_projection = nn.Linear(self.latent, 16, bias=False)
        nn.init.zeros_(self.style_projection.weight)

    def load_identity(self, state):
        result = self.load_state_dict(state, strict=False)
        if (
            set(result.missing_keys) != {"context_gate", "style_projection.weight"}
            or result.unexpected_keys
        ):
            raise ValueError("Only the new code projection and context gate may be absent")

    def writer_features(self, xy, style):
        if style is None or style.shape != (xy.shape[0], self.latent):
            raise ValueError("Explicit two-dimensional shared word code required")
        writer = self.writer[None] + self.style_projection(style)
        return writer[:, None, None].expand(*xy.shape[:-1], -1)


class MetricSemanticInk(ContextualSemanticInk):
    """Expose actual glyph scale/aspect to avoid contradictory normalized ink targets."""

    architecture = "semantic-metric-ink-field/1"

    def __init__(self):
        super().__init__()
        self.metric_projection = nn.Linear(2, 32, bias=False)
        nn.init.zeros_(self.metric_projection.weight)

    def load_identity(self, state):
        result = self.load_state_dict(state, strict=False)
        if (
            set(result.missing_keys) != {"context_gate", "metric_projection.weight"}
            or result.unexpected_keys
        ):
            raise ValueError("Only new metric projection and context gate may be absent")

    def character_features(self, ids, radii):
        width, height = radii.unbind(-1)
        metrics = torch.stack([(width / height).log(), (height / 24).log()], -1)
        return self.character(ids) + self.metric_projection(metrics.clamp(-5, 5))


class CoupledDistanceInk(AutoDecodedSemanticInk):
    """One learned distance field determines both ink occupancy and contour support."""

    architecture = "semantic-distance-word-ink/1"

    def forward(self, tokens, boxes, points, *, style=None):
        _, distance = super().forward(tokens, boxes, points, style=style)
        # signed_distance uses 8px units: sigmoid(16*d) has a 0.5px transition.
        # This is part of the trainable output contract, not image cleanup.
        return 16 * distance, distance


class WordLatentSemanticInk(ContextualSemanticInk):
    """A whole-word latent shared by all character fields; posterior is training-only.

    Its variation is not automatically personal style: content leakage and prior
    generation must be checked separately. The output is still offline ink.
    """

    architecture = "semantic-word-latent-ink/1"
    latent = 8

    def __init__(self):
        from research.corpus.reference_generator import block

        super().__init__()
        self.style_projection = nn.Linear(self.latent, 16, bias=False)
        nn.init.zeros_(self.style_projection.weight)
        self.posterior_encoder = nn.Sequential(
            nn.AvgPool2d(2),
            block(2, 16, 2),
            block(16, 32, 2),
            # Input is fixed 128x512 → 16x64 here. Fixed nonoverlapping pooling
            # has a deterministic CUDA backward, unlike adaptive_avg_pool2d.
            nn.AvgPool2d((8, 16)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 4, self.latent * 2),
        )

    def load_identity(self, state):
        result = self.load_state_dict(state, strict=False)
        allowed = {"context_gate", "style_projection.weight"} | {
            "posterior_encoder." + k for k in self.posterior_encoder.state_dict()
        }
        if set(result.missing_keys) != allowed or result.unexpected_keys:
            raise ValueError("Only new latent/context components may be absent")

    def writer_features(self, xy, style):
        if style is None or style.shape != (xy.shape[0], self.latent):
            raise ValueError("Explicit shared word latent required; no image input")
        writer = self.writer[None] + self.style_projection(style)
        return writer[:, None, None].expand(*xy.shape[:-1], -1)

    def posterior(self, target, semantic_base):
        if target.shape != semantic_base.shape or target.shape[1:] != (1, 128, 512):
            raise ValueError("Training-only whole-word target and neural base required")
        mu, logvar = self.posterior_encoder(torch.cat([target, semantic_base], 1)).chunk(2, 1)
        return mu, logvar.clamp(-8, 4)


class SpatialWordLatentInk(WordLatentSemanticInk):
    """Learned word/character-conditioned coordinates plus trainable neural ink.

    This is not bitmap warping: there is no input bitmap to resample. The bounded
    coordinate component is explicitly geometric; its success alone would not
    prove new stroke topology or connected-writing learning.
    """

    architecture = "semantic-spatial-word-latent-ink/1"

    def __init__(self):
        super().__init__()
        self.geometry = nn.Sequential(nn.Linear(8 + 32 + 24, 32), nn.SiLU(), nn.Linear(32, 5))
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)

    def load_identity(self, state):
        result = self.load_state_dict(state, strict=False)
        allowed = {"context_gate", "style_projection.weight"} | {
            name
            for name in self.state_dict()
            if name.startswith(("posterior_encoder.", "geometry."))
        }
        if set(result.missing_keys) != allowed or result.unexpected_keys:
            raise ValueError("Only new word variation components may be absent")

    def local_coordinates(self, xy, local, style):
        if style is None or style.shape != (xy.shape[0], self.latent):
            raise ValueError("Explicit word latent required")
        z = style[:, None, None].expand(*xy.shape[:-1], -1)
        condition = torch.cat(
            [
                z,
                self.character(local[..., 0]),
                self.neighbour(local[..., 1]),
                self.neighbour(local[..., 2]),
            ],
            -1,
        )
        value = self.geometry(condition).tanh()
        dx, dy, sx, sy, shear = value.unbind(-1)
        y = (xy[..., 1] - 0.4 * dy) * (-0.3 * sy).exp()
        x = (xy[..., 0] - 0.4 * dx - 0.25 * shear * y) * (-0.3 * sx).exp()
        return torch.stack([x, y], -1)
