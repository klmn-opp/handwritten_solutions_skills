# Handwritten Solutions v1.2.1

Platform admission correction for the v1.2.0 next-generation default backend.
No model weight, glyph, rendering algorithm or output claim changed.

## What changed

- The packaged author preset now admits the two actual GitHub Linux text
  environments: x86_64 with Python 3.12 and 3.13, Pillow 13.0.0, FriBidi 1.0.13.
- The maintainer Linux 3.12 environment with FriBidi 1.0.8 remains admitted.
- The public CI matrix now covers exactly the platforms this release verifies
  (Linux x86_64, Python 3.12 and 3.13). macOS and Windows remain outside the
  verified set; macOS runners currently expose Pillow BASIC layout, not RAQM.

The default backend still composes stable personal-CPU layout with the frozen
`nextgen-v1` English and mathematical glyph variation, deterministic ink/tone
correction and integer word spacing. Inference uses NumPy only; PyTorch is not
required. Unsupported glyphs, structure elements, failed guards and non-plain
paper keep the stable CPU path. Explicit `"backend": "font-cpu"` remains available.

## Verification scope

Actual installation, default rendering and pixel verification cover Linux x86_64
with Python 3.12 and 3.13. Ruled/grid backgrounds are still rejected for the
author preset because its final word gap requires semantic layout.

Host Agents remain responsible for reading, solving and independently checking the
mathematics. Rendering does not prove an answer.
