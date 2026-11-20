---
name: handwritten-solutions
description: Solve supplied mathematics, physics or statistics homework and produce a local CPU handwriting PDF with one page per top-level problem, preserving sources and verification records.
---

# Handwritten Solutions

Use the installed `hwsol`, or `uv run --offline --frozen --project <skill-directory>
hwsol ...` in an installed source checkout. This public package uses CPU fonts and
deterministic math layout. It includes no personal research model or neural-math
candidate. Do not claim that font construction trains a neural network.

1. Read the actual text, image or PDF. Preserve the original locally. For PDFs use
   `hwsol read-page --source <pdf> --page <n> --output <new.png>` and inspect every
   relevant page. Check signs, units, diagrams and subparts; request missing data
   when the problem cannot be determined reliably.
2. Inventory top-level problems in source order. Preserve printed numbers and
   subparts. Give repeated numbers separate IDs. Each top-level problem gets one
   page; its subparts stay together.
3. Solve in concise English with necessary steps, units and assumptions. Save
   one UTF-8 Markdown file per problem. Use ordinary text and regular-weight
   LaTeX, inline dollar delimiters and separate display blocks. Keep mathematical
   distinctions with supported notation; the [Job contract](references/jobs.md)
   describes the supported subset and explicit refusals.
4. Independently check the solution by substitution, numerical or symbolic
   recomputation, units or a separate derivation. Record only actual checks.
   Mark uncertainty `inconclusive` or `not_checked`; rework failed answers.
   A renderer check is not a proof of solution correctness.
5. Write the Job JSON, original inputs and solution files in one directory. Use
   relative local paths, an explicit Profile when requested, and record Agent
   authorship. [The guide](public_docs/GUIDE.md) and
   [five-problem example](examples/phase2/mixed/job.json) show the workflow.
   For an explicitly selected reviewed personal ink candidate, use
   `personal-cpu-experimental`, a pinned candidate Profile and its calibrated
   defaults. Read [candidate selection](references/jobs.md#reviewed-personal-cpu-candidate).
   This CPU mode is experimental, cannot advance latest and ships no private ink.
   For a locally qualified and explicitly published personal ink version, ordinary
   `font-cpu` Jobs can select that pinned Profile. Use baseline realism and inherit
   its reviewed render settings; do not bypass a stale-qualification rejection.
6. Run `hwsol render --manifest <job.json> --workspace <workspace>`. Read its
   immutable run receipt. Errors and unsupported formulas must remain explicit.
   For a changed solution create a new run; never delete required derivation to
   force a page fit. Never edit a sealed run or advance latest manually.
7. Run `hwsol verify --job <run> --pixels`. Inspect the pages against the source
   inventory for missing text, signs, mathematical structure, clipping and
   fallback. Deliver PDF, PNGs, editable answers and reports. Distinguish Agent
   solution verification from engineering checks and author style judgment.

Use [Studio and Profiles](public_docs/PROFILES.md) for user-authorized collection,
font construction, candidate evaluation, activation and rollback. Preserve raw
sources and labels; keep personal assets local. Normal rendering does not call
Studio. Unsupported personal glyphs may use a recorded font fallback.

For personal ink qualification, follow the [local lifecycle](public_docs/GUIDE.md#use-a-reviewed-personal-cpu-profile).
`profiles qualify-ink` validates an exact local evidence request: real supported
platform installations and complete Jobs, package/Profile identity, author review,
mathematical checks and current text capabilities. It neither publishes nor activates.
`profiles publish-ink` is a separate, explicitly authorized publication in the local
store; it uploads nothing and leaves the active default unchanged. Activation or
rollback requires its own user instruction. Imported ink archives always return to
the candidate area; never treat a ZIP, claimed approval flag or older package's
results as qualification of the current environment. Keep the bundled CPU font
usable while personal evidence is incomplete. These rules do not claim an untested
release candidate has passed, and they do not authorize sharing private ink.

[Experimental adapters and benchmark rules](public_docs/RESEARCH_BOUNDARIES.md)
are separate from normal output. No private model or author-specific research
archive ships here. Explicit user approval is required before publishing private
assets or changing a production default. Follow applicable course requirements.
