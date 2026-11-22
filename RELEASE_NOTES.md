# Handwritten Solutions v1.2.3

Personal-source enforcement and a radical-ownership fix. No model weight, glyph
array, checkpoint or neural forward pass changed.

## Fixes

- Overlapping glyph boxes (a radical sign and its radicand, for example) now keep
  per-item ownership from each placed tile. Previously a page crop let the radical
  claim every pixel in the shared box, leaving the radicand blank and raising an
  empty-message assertion. Empty tiles are skipped and future guard failures carry
  a diagnostic message.
- Every mathematical glyph whose CPU base was an `external-font` or `inherited-font`
  now uses a personal O atom. If the neural warp is rejected by topology, border or
  identity guards, the fitted personal O atom itself is used. The renderer refuses
  to leave a foreign math source. This covers `≠`, `%`, `ℏ`, `Γ`, `≡`, `⊆`, `∅`,
  `∪`, `∉`, `…`, `⇒`, `⇔` and the other O-covered labels.
- O-covered text marks such as `%` use personal O atoms; `:` and `'` use reviewed
  personal punctuation. A text mark without any personal sample (for example `;`)
  keeps the Profile glyph and is listed in `text_foreign_sources` instead of being
  silently claimed as personal handwriting.

## Verification

Seven-exercise replay: zero foreign math sources, zero text foreign sources, 2,411
English neural glyphs, 341 mathematical neural warps, 7 personal-atom fallbacks.
Ink/tone and word spacing preserve page contour counts. The real Mac and Linux
installations pass the default page render and `verify --pixels`.

Inference remains NumPy-only; PyTorch is not required. Explicit
`"backend": "font-cpu"` remains available. Windows remains outside the verified set.

Host Agents remain responsible for reading, solving and independently checking the
mathematics. Rendering does not prove an answer.
