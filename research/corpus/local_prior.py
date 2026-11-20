"""Train-derived latent neighbourhoods, never bitmap retrieval at generation."""

import math

import torch


def sample_neighborhood(centers, sigma, *, generator=None):
    """Draw near one observed joint code; unsupported axes remain fixed.

    Return the center index for auditable training supervision and provenance.
    This statistical sampler is not itself neural training or novel-style proof.
    """
    if centers.ndim != 2 or centers.shape[1] != 10 or len(centers) == 0:
        raise ValueError("Nonempty joint local code matrix N x 10 required")
    if not math.isfinite(sigma) or sigma < 0 or not torch.isfinite(centers).all():
        raise ValueError("Finite centers and nonnegative neighborhood scale required")
    centers = centers.detach()
    scale = centers.std(0) if len(centers) > 1 else torch.zeros_like(centers[0])
    index = int(torch.randint(len(centers), (1,), device=centers.device, generator=generator))
    noise = torch.randn(10, device=centers.device, dtype=centers.dtype, generator=generator)
    return centers[index] + sigma * scale * noise, index
