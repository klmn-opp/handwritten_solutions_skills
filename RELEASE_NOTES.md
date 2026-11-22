# Handwritten Solutions v1.2.2

macOS native admission for the next-generation default backend. No model weight,
glyph, layout or rendering setting changed from v1.2.1.

## What changed

- The packaged author preset now admits the actual Apple Silicon macOS text
  environments used for verification: arm64, Pillow 13.0.0 BASIC layout,
  FreeType 2.14.3, Python 3.12 and 3.13.
- Both runtimes completed a clean wheel install, `doctor`, the two-page default
  neural render and `verify --pixels` on the physical Mac.
- The public CI matrix covers Linux x86_64 and macOS arm64 on Python 3.12/3.13.
- Linux admission from v1.2.1 remains unchanged.
- Windows remains outside the verified set.

The default backend composes stable personal-CPU layout with the frozen
`nextgen-v1` English and mathematical glyph variation, deterministic ink/tone
correction and integer word spacing. Inference uses NumPy only; PyTorch is not
required. Unsupported glyphs, structure elements, failed guards and non-plain
paper keep the stable CPU path. Explicit `"backend": "font-cpu"` remains available.

## Verification scope

Actual installation, default rendering and pixel verification cover Linux x86_64
and macOS Apple Silicon (arm64) with Python 3.12 and 3.13. Ruled/grid backgrounds
are still rejected for the author preset because its final word gap requires
semantic layout.

Host Agents remain responsible for reading, solving and independently checking the
mathematics. Rendering does not prove an answer.
