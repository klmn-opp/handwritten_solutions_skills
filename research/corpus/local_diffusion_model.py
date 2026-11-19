"""Full-resolution stochastic ink, with local content conditions and no frozen codec."""

import torch
from torch import nn

from research.corpus.local_ink_model import LocalInk


class LocalInkDiffusion(LocalInk):
    architecture = "local-full-resolution-ink-diffusion/1"

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU())

    def forward(self, noisy: torch.Tensor, reference: torch.Tensor, time: torch.Tensor):
        if noisy.shape != reference.shape or noisy.ndim != 4 or noisy.shape[1] != 1:
            raise ValueError("Matching noisy ink and content condition required")
        if time.shape != (len(noisy),):
            raise ValueError("One diffusion time per sample required")
        field = (2 * time - 1)[:, None, None, None].expand_as(noisy)
        h = self.stem(torch.cat([noisy, 2 * reference - 1, field], 1))
        for block in self.blocks:
            h = h + 0.2 * block(h)
        return self.output(h)


def noise_schedule(time: torch.Tensor):
    angle = time[:, None, None, None] * (torch.pi / 2)
    return angle.cos(), angle.sin()


@torch.inference_mode()
def generate_diffusion(model, reference: torch.Tensor, noise: torch.Tensor, steps: int = 64):
    if reference.shape != noise.shape or steps < 2:
        raise ValueError("Independent matching noise and at least two steps required")
    current = noise.clone()
    times = torch.linspace(1, 0, steps + 1, device=noise.device)
    for i in range(steps):
        time = times[i].expand(len(noise))
        alpha, sigma = noise_schedule(time)
        velocity = model(current, reference, time)
        clean = (alpha * current - sigma * velocity).clamp(-1, 1)
        epsilon = sigma * current + alpha * velocity
        next_alpha, next_sigma = noise_schedule(times[i + 1].expand(len(noise)))
        current = next_alpha * clean + next_sigma * epsilon
    return (current + 1) / 2
