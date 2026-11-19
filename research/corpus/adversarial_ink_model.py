"""Local real-ink distribution experiment; not RL or a preference reward model."""

import torch
import torch.nn.functional as F
from torch import nn


class InkPatchCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 24, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(24, 32, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, ink: torch.Tensor) -> torch.Tensor:
        return self.net(ink)


def structure_loss(
    ink: torch.Tensor, reference: torch.Tensor, *, relaxed: bool = False
) -> tuple[torch.Tensor, dict]:
    if ink.shape != reference.shape or ink.ndim != 4:
        raise ValueError("Matching local ink and content guide required")
    structure = ink.new_zeros(())
    scales = [(16, 1.0), (32, 1.0)] if relaxed else [(4, 1.0), (8, 1.0), (16, 0.5)]
    for scale, weight in scales:
        source = F.avg_pool2d(reference, scale)
        prediction = F.avg_pool2d(ink, scale)
        presence = F.max_pool2d(reference, scale)
        structure = structure + weight * ((prediction - source).abs() * (1 + 4 * presence)).mean()
    radius = 12 if relaxed else 4
    support = (F.max_pool2d(reference, radius * 2 + 1, 1, radius) > 0.05).to(ink.dtype)
    outside = (ink * (1 - support)).mean()
    entropy = (ink * (1 - ink)).mean()
    return 0.7 * structure + 2 * outside + 0.2 * entropy, {
        "structure": structure,
        "outside": outside,
        "entropy": entropy,
    }


def word_patches(
    images: torch.Tensor, centers: list[tuple[int, int, int]], radius: int
) -> torch.Tensor:
    padded = F.pad(images, (radius, radius, radius, radius))
    return torch.stack(
        [padded[i, :, y : y + 2 * radius, x : x + 2 * radius] for i, y, x in centers]
    )


@torch.inference_mode()
def generate_local(model, reference: torch.Tensor) -> torch.Tensor:
    return model(F.pad(reference, (32, 32, 32, 32))).sigmoid()[..., 32:-32, 32:-32]
