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

The optional `personal-cpu-experimental` backend reads a pinned private candidate
Profile containing reviewed outlines and geometry bindings. It uses baseline
realism, raw metrics, personal math strokes and one fixed page-layout attempt.
Every rejected replacement retains a recorded reason; unsupported structure or
overflow rejects the page. It cannot activate a Profile or advance formal latest.
It uses no Torch, research modules, old experiment directories or cached word images.
See [candidate selection and portability](../references/jobs.md#reviewed-personal-cpu-candidate).

The `personal-neural@1` backend runs the frozen `nextgen-v1` model after the stable
personal CPU composition. It replaces only eligible English letters and ordinary
mathematical atoms. Fraction bars, radicals, matrices, structure glyphs, unsupported
symbols and rejected candidates keep the stable CPU glyph. A candidate is rejected
on a new dark contact, changed components/holes, page-boundary ink, or a failed
single-symbol identity check. The deterministic ink/tone stage and integer word
spacing must preserve page contour counts. Non-plain paper delegates to the stable
CPU path; the bundled author preset currently rejects ruled/grid paper because its
final word gap requires semantic layout.
Inference uses NumPy only; no parameter is updated at render time.
See [the model page](../public_docs/NEURAL_MODEL.md).
