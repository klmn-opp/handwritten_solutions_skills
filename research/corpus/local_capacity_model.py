"""Controlled wider local field initialization; this operation is NOT training."""

import torch

from research.corpus.semantic_ink_model import ContextIndependentLocalInk


@torch.no_grad()
def widen_local_field(source, *, seed=3719, noise_ratio=0.01):
    """Duplicate hidden units, splitting outgoing weights with zero-sum noise.

    Initial functions agree up to floating-point summation differences. Paired
    perturbations break neuron symmetry so subsequent neural updates can use the
    added capacity. No source image, font contour or generated pixel is copied.
    """
    if type(source) is not ContextIndependentLocalInk or source.hidden_width != 128:
        raise ValueError("Registered conversion requires the 128-wide local field")
    if not 0 < noise_ratio <= 0.1:
        raise ValueError("Small nonzero symmetry-breaking ratio required")
    parameter = next(source.parameters())
    target = ContextIndependentLocalInk(hidden_width=256).to(
        device=parameter.device, dtype=parameter.dtype
    )
    old, new = source.state_dict(), target.state_dict()
    generator = torch.Generator(device=parameter.device).manual_seed(seed)

    def split_columns(weight):
        result = weight.repeat_interleave(2, dim=1) / 2
        noise = torch.randn(
            weight.shape, device=weight.device, dtype=weight.dtype, generator=generator
        ) * (noise_ratio * weight.std())
        return result + torch.stack([noise, -noise], -1).flatten(-2)

    for name, value in old.items():
        if not name.startswith(("field.", "field_modulation.")):
            new[name] = value.clone()
    for layer in [0, 2, 4]:
        weight = old[f"field.{layer}.weight"].repeat_interleave(2, dim=0)
        new[f"field.{layer}.weight"] = split_columns(weight) if layer else weight
        new[f"field.{layer}.bias"] = old[f"field.{layer}.bias"].repeat_interleave(2)
    new["field.6.weight"] = split_columns(old["field.6.weight"])
    new["field.6.bias"] = old["field.6.bias"].clone()
    new["field_modulation.weight"] = (
        old["field_modulation.weight"]
        .reshape(3, 2, 128, 10)
        .repeat_interleave(2, dim=2)
        .reshape(1536, 10)
    )
    new["field_modulation.bias"] = (
        old["field_modulation.bias"].reshape(3, 2, 128).repeat_interleave(2, dim=2).reshape(1536)
    )
    target.load_state_dict(new)
    return target
