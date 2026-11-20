# Rendering, layout and glyph provenance

The CPU backend uses Handnote and MathText to position glyphs and mathematical
structures. It accepts a restricted, checked LaTeX subset. Unsupported commands,
untraceable formula fallback, page overflow and unreadable output must be reported.
Do not silently simplify a submitted formula or treat zero fallback on a fixed
test set as universal symbol coverage.

`realism` selects `baseline`, `layout-enhanced`, `stroke-enhanced`, or `combined`.
The resolved mode is recorded in the manifest. Optional rendering factors are
`glyph_metrics` (`raw`, `normalized`, `contextual`), `semantic_layout` and
`math_strokes` (`personal`, `off`). Their availability does not prove that any
particular Profile improves personal similarity. Retain the seed and Profile
digest for controlled comparisons and historical replays.

Baseline is suitable for a reproducible first job. Contextual geometry and nib
variation remain font/layout operations. They do not create a trained continuous
handwriting model. Personal structure strokes require appropriate registered
Profile assets; unsupported roles preserve explicit fallback.

Reports distinguish collected personal glyphs, inherited or external font glyphs,
and structural geometry. Preserve those reports with the PDF. Matching parser
traces and page pixels can show a reproducible rendering without proving the
mathematics, character recognizability or author preference.

CPU renderer policies are optional experimental distributions over finite
rendering actions. `hwsol lab rl --help` exposes their workflow. A policy must
match its Profile and rendering fingerprint; it is embedded in `render.policy`
and requires `stroke-enhanced` or `combined`. This is not neural generator RL.
Do not apply old ratings to new images or activate a policy without evaluation.

See [the Job contract](../references/jobs.md) and [workspace records](WORKSPACE.md).
