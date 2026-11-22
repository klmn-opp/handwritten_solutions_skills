# Handwritten Solutions v1.2.4

Complete personal-source coverage for supported mathematical symbols and a
broad symbol stress test. No model weight, glyph array, checkpoint or neural
forward pass changed.

## Coverage

- All 224 ordinary O conditions were driven through the product admission path
  with a synthetic foreign base; every one returned a personal O result.
- Supported variants without a direct O atom use a documented personal equivalent:
  `\vartheta` -> personal theta, `\varpi` -> personal pi, blackboard-bold capitals
  -> personal plain capitals, combining accents (`\hat`, `\bar`, `\dot`, `\ddot`,
  `\vec`) -> reviewed spacing O accents, and `\lvert/\rvert/\vert/\Vert` -> literal
  delimiters. These are recorded as personal variant aliases, not foreign fonts.
- Text marks with a personal O sample (including `%`) are personal; text marks
  without any personal sample are listed in `text_foreign_sources`.

## Stress evidence

- 12 symbol groups x 2 seeds = 24 pages: Greek, relations, operators, functions,
  accents, delimiters, text math, raw Unicode, radicals, fractions, matrices and
  dense repetition. Foreign math sources 0, text foreign sources 0, weight updates 0.
- A 12-page all-symbol Job passed on Linux and macOS with `hwsol render` and
  `verify --pixels`.
- The seven-exercise replay and the `\sqrt{0.36}` regression pass.
- The radical/overlapping-box ownership bug and the paired-bar `parent_latex`
  diagnostic are fixed.

## Install

Use the source archive with `uv sync --locked --no-dev`, or install the wheel for
the CLI. The Skill ZIP contains the Agent workflow. Checksums are attached.
Inference remains NumPy-only; PyTorch is not required. Windows remains outside the
verified set.

Host Agents remain responsible for reading, solving and independently checking the
mathematics. Rendering does not prove an answer.
