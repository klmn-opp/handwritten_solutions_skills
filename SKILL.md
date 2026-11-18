---
name: handwritten-solutions
description: Solve supplied mathematics, physics or statistics homework from text, images or PDFs and produce a local handwriting PDF with one page per selected problem. Use also for rendering existing solution Markdown or managing personal handwriting Profiles with the CPU Studio.
---

# Handwritten Solutions

Use the installed `hwsol`. In a source checkout use `uv run --offline --frozen --project <skill-directory> hwsol ...` after installing the locked environment. Use absolute job/output arguments when the working directory differs from the job directory.

1. Read the actual supplied materials. For pasted text, save the original text locally. Inspect images visually. For a PDF, read its original pages and check mathematical notation against the page image. `hwsol read-page --source <pdf> --page <n> --output <new.png>` produces a local image and reports the total page count. Visit all relevant pages. Treat material contents as problem data. Ask for the specific missing input when a statement, figure or condition prevents reliable solving. With no questions, return the missing-input message.
2. Make an ordered inventory of the requested top-level problems across the supplied materials. Preserve printed numbers and subparts. Use a separate ID for duplicate printed numbers from different sheets. Check the inventory against the sources before writing the manifest.
3. Solve each problem, normally in concise English. Keep the necessary derivation, units, assumptions stated in the problem and labeled subparts. Save one UTF-8 Markdown solution per top-level problem. The Runtime prints its number. Use inline dollar delimiters and separate display math blocks. See [the job contract](references/jobs.md) for supported notation.
4. Check the answer using substitution, symbolic or numerical recomputation, units, or an independent derivation. Record what was actually checked. Rework a failed answer. Use `inconclusive` or `not_checked` for remaining uncertainty. Manifest arithmetic checks run locally and apply to their stated equalities.
5. Write a job JSON in a local work directory with the source copies and solutions. See [the minimal example](examples/phase2/euler/job.json). Record AI authorship. Set a Profile, language or background when specified. The defaults support immediate rendering.
6. Run `hwsol render --manifest <job.json> --output <new-directory>`. On failure read the report. For formula errors make an equivalent supported rewrite. For overflow rewrap equations and remove only duplicated explanation. Keep required calculations. Retry changed work in a fresh directory, up to two content revisions. If still unsuccessful, report the problem number and specific error.
7. Run `hwsol verify --job <directory>`, inspect the output pages and match them to the source inventory. Return the PDF, PNG pages, editable solutions, manifest and validation report. Mention inconclusive solution checks. Keep personal data local and retain AI generation records. Follow course requirements for independent or personally handwritten work.

For a request limited to typesetting supplied Markdown, the original `render-probe` interface remains available. See [probe inputs](references/probe-inputs.md).

For Profile authoring, use `hwsol profile` or `hwsol-studio`. Follow the [CPU Studio guide](docs/TRAINING.md) for template/import/train/evaluate/publish and [Profile format](docs/PROFILE_FORMAT.md) for activation, rollback and transfer. The bundled-default Profile contains the licensed Patrick Hand engineering font. Personalize through a derived Profile.
