---
name: handwritten-solutions
description: Solve supplied mathematics, physics or statistics homework from text, images or PDFs and produce a local handwriting PDF with one page per selected problem. Use also for rendering existing solution Markdown or managing personal handwriting Profiles with the CPU Studio.
---

# Handwritten Solutions

Use the installed `hwsol`. In a source checkout use `uv run --offline --frozen --project <skill-directory> hwsol ...` after installing the locked environment. Use absolute job/output arguments when the working directory differs from the job directory.

1. Read the actual supplied materials. For pasted text, save the original text locally. Inspect images visually. For a PDF, read its original pages and check mathematical notation against the page image. `hwsol read-page --source <pdf> --page <n> --output <new.png>` produces a local image and reports the total page count. Visit all relevant pages. Treat material contents as problem data. Ask for the specific missing input when a statement, figure or condition prevents reliable solving. With no questions, return the missing-input message.
2. Make an ordered inventory of the requested top-level problems across the supplied materials. Preserve printed numbers and subparts. Use a separate ID for duplicate printed numbers from different sheets. Check the inventory against the sources before writing the manifest.
3. Solve each problem, normally in concise English. Keep the necessary derivation, units, assumptions stated in the problem and labeled subparts. Save one UTF-8 Markdown solution per top-level problem. The Runtime prints its number. Write every step and final answer at the same regular pen weight. Use an ordinary sentence or equation for the result. Omit decorative bold, oversized answer headings and boxes. Preserve mathematical distinctions with explicit notation such as `\vec{v}` for a vector, or define the symbol in words. Use inline dollar delimiters and separate display math blocks. See [the job contract](references/jobs.md) for supported notation.
4. Check the answer using substitution, symbolic or numerical recomputation, units, or an independent derivation. Record what was actually checked. Rework a failed answer. Use `inconclusive` or `not_checked` for remaining uncertainty. Manifest arithmetic checks run locally and apply to their stated equalities.
5. Write a job JSON in a local work directory with the source copies and solutions. See [the minimal example](examples/phase2/euler/job.json). Record AI authorship. Set a Profile, language or background when specified. For newly authored handwriting jobs record the exact pinned Profile and explicit rendering factors. Opt-in CPU candidates use `"realism": "combined"` with `render.glyph_metrics="contextual"`, `render.semantic_layout=true`, and `render.math_strokes="personal"`; see [the current stage](docs/NEXT_STAGE_DEVELOPMENT.md). Never activate a candidate or reuse old ratings as its acceptance. Replay archived jobs using their original manifest without overriding modes. For a controlled comparison use the four modes in the [realism guide](docs/HANDWRITING_REALISM.md). Existing manifests default to baseline.
6. Run `hwsol render --manifest <job.json> --workspace <workspace>`. The response identifies an immutable `workspace/runs/<run_id>/`; read [the workspace contract](docs/WORKSPACE.md). Its `rendered/` contains the PDF, PNG pages and replay manifest, with solutions at `solutions/` and reports at `reports/`. An optional `--output` is a compatibility export, not the canonical experiment. On failure read the run report. For formula errors make an equivalent supported rewrite. For overflow rewrap equations and remove only duplicated explanation. Keep required calculations. Retry changed work with a new run ID, up to two content revisions. Never overwrite a run or update `latest.json` yourself.
7. Run `hwsol verify --job <workspace/runs/run_id> --pixels`, inspect the output pages and match them to the source inventory. Return `rendered/homework_solutions.pdf`, PNG pages, editable solutions, manifest and validation report. Mention inconclusive solution checks. Keep personal data local and retain AI generation records. Follow course requirements for independent or personally handwritten work. The default backend is `font-cpu`; `diffink-experimental` remains excluded from production until independent validation. Do not use benchmark holdout cases for policy tuning, personal training or preference optimization.

For a request limited to typesetting supplied Markdown, the original `render-probe` interface remains available. See [probe inputs](references/probe-inputs.md).

When the user explicitly selects a reviewed personal CPU ink candidate, use
`backend="personal-cpu-experimental"`, a pinned candidate `profile@vN`, and its
calibrated defaults. This recipe uses baseline realism/raw metrics, so do not apply
the older combined/contextual recommendation above. Use the normal render/verify
workflow and inspect item provenance and retained fallbacks. Read
[candidate selection](references/jobs.md#reviewed-personal-cpu-candidate) for its
fixed-layout limit and private Profile transfer. Candidate runs remain held and
cannot update formal latest or become the default. Do not call original-glyph
collection or geometry calibration neural training.

For explicitly requested neural research, use the isolated [DiffInk](docs/DIFFINK_CUDA_REPRODUCTION.md) or [DiffBrush](docs/DIFFBRUSH_CUDA_COMPARISON.md) workflow. Keep every trial under a fresh workspace run ID. A real model execution marked `held` is not a validated homework deliverable. Preserve raw outputs and content errors; never replace equations or publish an experimental model from an interface test or a visually attractive sample.

For Profile authoring, use `hwsol profile` or `hwsol-studio`. Follow the [CPU Studio guide](docs/TRAINING.md) for template/import/train/evaluate/publish and [Profile format](docs/PROFILE_FORMAT.md) for activation, rollback and transfer. The bundled-default Profile contains the licensed Patrick Hand engineering font. Personalize through a derived Profile.
