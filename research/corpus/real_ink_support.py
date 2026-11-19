"""Optional Torch training loss: nearest observed real shape, never output retrieval."""

import torch
from torch.nn import functional as F


def nearest_real_loss(logits, variants):
    """One P-vector prediction against K valid real P-vector observations.

    Select a whole observation for the loss, not a per-pixel union/average that
    could describe no real glyph. The selected image is never the model output.
    """
    if logits.ndim != 1 or variants.ndim != 2 or variants.shape[1] != logits.shape[0]:
        raise ValueError("One query vector and complete matching observed variants required")
    if not len(variants) or not torch.isfinite(variants).all():
        raise ValueError("Finite nonempty observed real support required")
    loss = F.binary_cross_entropy_with_logits(
        logits[None].expand_as(variants), variants, reduction="none"
    )
    return (loss * (1 + 3 * variants)).mean(1).min()
