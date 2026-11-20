# Job contract v1

Create a job directory containing the original material, one Markdown solution per selected top-level problem, and an input manifest. Relative paths use forward slashes and stay within this directory. Example files are in [examples/phase2](../examples/phase2). A schema for editors is in [job.schema.json](job.schema.json).

| Field | Meaning |
|---|---|
| schema_version, generated_by | Use 1 and the actual authoring agent/AI generation record. |
| problems | Ordered selection. Each row needs a printed number, a transcribed statement, one or more sources and a solution path. Use distinct optional `id` values when the same number occurs on multiple sheets. |
| sources | Each entry has a local path and optional one-based page, location text and SHA-256. Supported inputs are UTF-8 TXT/MD, PDF and common raster images. Set `complete: false` on the problem when essential information is missing. |
| verification | Agent status is not_checked, passed, failed or inconclusive. Supply evidence for reported checks. Optional `checks` pairs contain a numeric lhs and rhs. Exact rational evaluation accepts decimal or integer literals, parentheses, +, -, *, / and bounded integer powers using **. A failed check stops publication. |
| language | Describes the Agent's solution language, default en. The Agent writes the requested language. The selected font must cover its prose. |
| profile, profile_sha256 | Override the active Profile by name or pinned reference. An optional hash verifies the exact snapshot. The delivered manifest pins both. |
| backend | `font-cpu` is the default. `personal-cpu-experimental` explicitly selects a locally installed, pinned candidate ink Profile; see below. `diffink-experimental` remains reserved and fails closed. Mathematics remains deterministic. |
| realism | Job-level baseline (default), layout-enhanced, stroke-enhanced, or combined. The optional CLI override `--realism` is saved in the delivered manifest. See [CPU realism](../docs/HANDWRITING_REALISM.md). |
| render | RenderConfig overrides such as dpi, ink, seed, page_width_mm, page_height_mm, margin_mm, font_size_pt and line_spacing. The default Profile uses A4, 150 DPI and an 18 mm margin. Set dpi to 300 for high-resolution PNG. |
| render.policy | Optional frozen CPU renderer policy, bound to the Profile digest. Embed the complete policy.json object; a filesystem path is not accepted. Requires stroke-enhanced or combined. Actions and probabilities remain in the archived manifest for replay. See [RL experiments](../docs/RL_RENDERING.md). |
| background | Job default or per-problem override. Choose plain, lined, grid, dots, image or pdf as kind. For custom paper include path, optional one-based PDF page and fit (contain, cover or stretch). |

Paper precedence is problem, job, user setting, then the Profile's render.paper preset (bundled default uses plain). To set a persistent default, preserve existing keys in the data directory's settings.json and set its background object, for example `{"kind":"lined"}`. A custom paper file in persistent settings may be an absolute path or a path relative to the user data directory. A job-specific paper file is relative to the manifest. The delivered job includes the used assets and resolved per-problem choices. A PDF background is rasterized at page scale and embedded below vector ink. Background content is outside the solution ink bounds check.

Use Markdown paragraphs, headings, and lists. Inline math uses single dollar delimiters. Display math starts on its own line with double dollars. Use separate display blocks for successive derivations. Standalone matrices support matrix, bmatrix, pmatrix and vmatrix. The exact command whitelist is `COMMANDS` in [layout.py](../src/hwsol/jobs/layout.py). Supported groups include fractions, scripts, roots, integrals, sums, limits, vectors, Greek letters, statistics, relations and upright units. Use e.g. `\mathrm{m/s^2}`. Explicit page breaks in an answer fail. Text and one-character matrix entries need Profile coverage. The report lists missing Profile coverage and the fallback fonts observed in recorded math glyphs.

The homework harness applies the same pen treatment to derivations and results in all four modes. It fills the original or selected variant outline once, uses the body size for headings and ordinary inline and display formulas, and keeps scripts, fraction rules and matrix brackets. Paired Markdown strong markers around prose or inline formulas render at regular weight. TeX bold commands and Unicode mathematical bold characters require an explicit rewrite. Remove decorative emphasis, use `\vec{v}` for a vector, or define the intended matrix or tensor in words. The archived Markdown retains the submitted source. The per-page handwriting report records `glyph_ink` as `fill-only`. Verification matches that policy to the PDF page and inspects the actual painting operators in its outline forms.

Ordinary math letters, digits, fraction numerators/denominators and matrix cells share the prose em. The renderer uses MathText's native display-style fraction layout internally and keeps the original LaTeX in the archived source and report. Scripts retain their structural reduction. Wide handwritten math alternates fit the available width independently of their vertical metrics, so width adaptation does not shrink their height. Matrix cells expand to fit larger expressions. The handwriting report records this as `math_size: body-em/full-size-fractions`; this is a font-size policy, not a claim that different letters have identical ink heights.

CLI output is canonical under `workspace/runs/<run_id>/rendered/`; see [workspace contract](../docs/WORKSPACE.md). An optional legacy `--output` export must be empty or new. The generator stages all artifacts, checks the placed vector ink and page count, then merges and rasterizes the final PDF. Automatic retries remove explicit blank blocks, split eligible equality chains, and adjust margins, size and spacing within the documented Runtime bounds. An agent can further simplify repeated prose while preserving the derivation. Six layout configurations are the maximum per render. Manually specified smaller dimensions or type sizes are respected.

A success contains homework_solutions.pdf, pages/NN.png, solutions/NN.md, inputs/, optional paper/, manifest.json and validation_report.json. Every PDF page carries the problem ID, printed label and solution digest. The report records order, layout attempts, mathematical evidence and file digests. `hwsol verify --job DIR` checks the delivered files locally. The report records Agent checks separately from Runtime arithmetic and file integrity. For a source edit, render the delivered manifest to a new directory. Keep the original solution and grading context together when sharing an edited job. Originals in inputs/ can contain other questions from the supplied material.

After a failed build, the output contains validation_report.json. Read its problem, type, error and any attempts. Existing nonempty destinations are preserved. If a process is killed, a sibling .NAME.hwsol.lock may remain. Confirm its recorded PID has stopped before removing the stale lock.

## Reviewed personal CPU candidate

An audited private ink bundle can be selected with
`"backend":"personal-cpu-experimental"` and an explicit `"profile":"name@vN"`.
Resolution uses the local **candidates** area, not the active/default version.
The recipe requires `realism="baseline"`, raw glyph metrics, personal math strokes
and no renderer policy. Its own Profile supplies the calibrated rendering defaults.
This is a CPU outline/geometry recipe, not a neural generator. No private ink ships
with the public package, and arbitrary newly collected scans are not automatically
admitted to this calibrated recipe.

Use the ordinary render and verify commands. The same ordered rules apply to all
pages and content; no page-number dispatch or pre-rendered word cache is used.
Each visible item retains source provenance and rejected fits/fallback reasons.
Unsupported notation and page overflow reject the page explicitly. The current
candidate uses one fixed layout attempt, without the default backend's shrink or
rewrap retries. The Agent may author an equivalent supported solution in a new run;
it must not silently change a submitted formula or delete required reasoning.

When the Agent's own answer uses unsupported notation, keep the failed Job and
validation error, then author an explicitly equivalent answer in a new Job. Record
the source diff and recheck the mathematics before rendering. For example, the
current candidate rejects `\det` and `\widehat`; a named determinant written as
`\mathrm{det}(A)` and a single-letter `\hat{\mu}` can express the intended operations in
supported notation. This is an answer-authoring recovery, not a parser expansion
or a successful first attempt. Never remove a required step merely to fit a page.

Successful artifact verification does not publish this candidate: its workspace
run is `held`, `deliverable=false`, and `latest.json` remains unchanged. Profile
activation and the ordinary Studio publication route reject experimental ink.
The CLI response's `status` is the workspace admission status; a separate
`result_status="passed"` reports successful engineering validation. A passing
render result cannot override the run's admission status or identity.
Candidate portability uses `hwsol profiles export name@vN --candidate --output FILE`
and `hwsol profiles import --source FILE`. Import validates the hashed dependency
graph and installs it as a candidate without activation. The private archive contains
source crops and provenance and is not an authorized public distribution.
