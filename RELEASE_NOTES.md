# Handwritten Solutions v1.2.0

The author-accepted next-generation glyph model is now the default backend for the
bundled personal Profile. Stable CPU layout still decides mathematical structure,
slots, line breaks and page bounds; the frozen model changes only English and
mathematical glyph shape, followed by deterministic ink/tone correction and integer
word spacing.

## What changed

- New `personal-neural@1` backend; a Job that omits `backend` and resolves to
  `author-default@v1` uses it automatically on plain paper.
- The two accepted endpoints ship as value-identical float32 arrays plus the
  minimal 1,706 inference glyphs. No optimizer state, raw scan or full training
  corpus is included.
- A maintainer audit compared all 72 arrays with the source checkpoints byte for
  byte; no weight was trained, edited or selected during publication.
- Inference uses NumPy only. PyTorch is not installed or required.
- Unsupported symbols, structure elements, failed guards and identity mismatches
  keep the stable CPU glyph. Explicit `"backend": "font-cpu"` preserves the
  previous deterministic path. Patrick Hand and user-created Profiles remain CPU.
- New model page and seven-exercise demo are included in the source package.

## Verification scope

Actual installation, default rendering, fallback and pixel verification for this
version cover **Linux x86_64 with Python 3.12**. macOS and Windows are not
re-verified in 1.2.0 and the packaged author-preset admission will not activate on
unverified environments. Ruled/grid backgrounds are currently rejected for the
author preset because its final word gap requires semantic layout.

## Install

Use the source archive with `uv sync --locked --no-dev`, or install the wheel for
the CLI. The Skill ZIP contains the Agent workflow. Checksums are attached.

Host Agents remain responsible for reading, solving and independently checking the
mathematics. Rendering does not prove an answer.
