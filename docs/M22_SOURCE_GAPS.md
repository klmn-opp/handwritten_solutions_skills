# M2.2 — remaining source roles and continuous-ink readiness

2026-11-20. This increment preserves production, original labels/scans and all old
Profiles. It performs source review and one bounded CPU candidate addition; no
neural updates. Author style approval and actual macOS validation are separate gates.

## Actual historical search before source selection

[Audit b8673da8](../workspace/runs/20261120T154606311013Z-m22-math-source-gap-audit-b8673da8/run.json)
uses `research.corpus_access.catalogue` on the two established historical catalogues.
The metadata-first source/name/split/ancestry closure excludes sealed and heldout
sources before label retrieval. First catalogue exposes12 allowed documents and239
allowed samples; second exposes3 documents and0 samples. Eleven locator hits are
recorded:8 original sample labels and3 recognition records, not11 independent glyphs.
No match for double-struckZ was found in this bounded search. That does **not** prove
absence from all original historical pages or authorize scanning sealed sources.

All8 authentic historical universal-quantifier line crops were visually inspected,
not merely matched by OCR. They share sourceSHA
`59963b23ead1a7b77f1e39859ee93b7384f73d321ce73812553c5dc83456ef69`.
Their original is a native144DPI image, not a PDF; the forall shapes are roughly
14–20pixels wide and some are isolatable. [Original contact sheet](../workspace/analysis/m22-historical-quantifier-review/original-crops.png),
[one native historical candidate](../workspace/analysis/m22-historical-quantifier-review/historical-universal-native144.png)
and `historical-universal-candidate.json` preserve actual original pixels, SHA,
parent and source coordinates. They are not upsampled and falsely called300DPI ink.
Original metadata labels remain untouched; source review is a new separate record.

For the bounded rendering increment, the already supplied SessionO original is
selected because its native300DPI stroke is43pixels wide and has verified PDF↔raster
transforms. Historical evidence exists; low-resolution source quality, not supposed
source absence, motivates this selection. No new author collection is requested.

## Role decisions

| Role | Evidence and decision |
| --- | --- |
| ∀ | O/P10/R01 has one clear inverted-A original with crossbar. Exactcrop[252,707,311,785],8px original ink margin each edge; original RGB unchanged. Root independently viewed and approved source. |
| ℤ | O/P09/R06 actually contains ordinary single-lineN/Z/Q/R/C. Preserve this actual transcription; never invent a second stroke or relabel plainZ as a new double-struck shape. Explicit fallback remains. Historical bounded index search has no reliable corresponding hit. |
| DenseΦ | Reliable sources already exist. Uniform whole-source scaling by real bowl height preserves the unusually long stem; a particular fraction slot has only0.157286px clearance to its bar versus required0.60px. This is a placement constraint, not missing source or excessive source ink. Retain prior glyph and explicit refusal. |
| Matrix-spanning brace, widehat, ell | Parser/layout capability constraints, not solved by another raster glyph. Keep explicit unsupported-structure rejection until a separately verified structural implementation exists. |

Next densePhi investigation should measure slot reserve and baseline/structure-aware
placement across a fixed density set. Do not thin, erase or cut the author's stem;
do not relax the minimum clearance to make a report pass. A safe bounded layout
change needs mathematical structure preservation and unchanged unaffected text.

## Isolated candidate and proof

Implementation: `research/m22_source_audit.py` and
`research/m22_universal_candidate.py`. Source review/extraction uses the existing
`research.session_o_sources` workflow, retaining the original PDF,300DPI crop,
context, sourceSHA and all coordinate transforms.

Final source extracte50445ba → admittedbankca06ea37. Private asset increment
[018681e8](../workspace/runs/20261120T154816694269Z-m22-universal-private-v3-018681e8/run.json)
is `personal-cpu-candidate@v3`, parentv2. It appends only the approved∀ source and
explicit role; old36sources and other assets remain unchanged. This source increment
does not change the renderer; the separate contact correction below changes admission.
The earlier1a3f40f0 draft is retained but superseded because copied creation metadata
still described its parent. Finalv3 updates creation time and build lineage explicitly.
Neither candidate is published or activated.

Installed public CLI proof:
[2e24b3c7](../workspace/runs/20261120T154844926147Z-m22-universal-placement-2e24b3c7/run.json)
uses exact existing stress input and installed8da224f4 outside checkout in a new
isolated store. Bothv2/v3 use the same source, layout and seed; original input is not
rewritten. It checks actualPDF/PNG, semantic inventory and non-target geometry/source
identity. Its final report, not the source approval, determines placement acceptance.
Source approval alone is never production admission or author style judgment.

The actual v3 stress page passes placement and retains the explicit ℤ fallback:
2font fallbacks become1, with only∀ replaced. Independent pixel comparison in
`workspace/analysis/m22-universal-final-comparison/FINAL_REVIEW.json` finds440 changed
pixel positions, all inside the old/new∀ bounding union plus3px antialias margin;
outside that region pixels are exact. The formula pair is saved alongside the receipt.

## New E2E contact: bounding boxes do not prove ink separation

The unchanged fourth new problem, “Coaxial electrostatic field”, revealed a real
proposal-admission omission. In its title, personal d0 and preceding l have no path
contact. Selecting the recorded lower-bowl d1 stays within the same outer bounds
`[721.9,243.11,739.564,276.322]` but moves occupied ink into the l contour. The final
whole-chain gate correctly rejected the original four-page Job. This was unrelated
to missing symbol data, text accuracy, or ∀ admission.

Before diagnostic:b1be354b; after diagnostic:e508eba1. Both are fresh runs with the
actual unchanged fourth input, seed1909, original candidatev2 and exact kernel source
snapshot. They show contactFalse→True before correction andFalse→False after. The
three-way original/proposal/fixed title comparison and record equality check are in
`workspace/analysis/m22-cpu-contact-review/`. All original d0 provenance fields remain
exact, with only a rejection receipt added.

`CpuGlyphFeedback.apply` now checks every proposed replacement using existing
filled-outline `contacts()`, and restores its original item when a new contact would
be introduced. It records attempted variant/bounds/treatment, offending neighbor and
original-source retention. No layout shift, ink shrink, erasure, title rule or relaxed
final gate is involved. The existing math wrapper and final whole-chain gate remain.

Full original four-problem Job07f2da79 passes with0font fallbacks and pixel verification
using the editable current source; it is **not** exact-commit installed-package evidence.
Root will freeze and verify a new wheel separately. Source-bound24-page regression is
recorded separately before declaring preservation of old accepted output.

## New user M3: continuous handwriting (old master M4)

The user's new milestone numbering changes names, not historical experiment facts.
The [context source review](M4_CONTEXT_DATA_AUDIT.md) and
[alignment pilot](M4_ALIGNMENT_PILOT.md) remain the starting evidence. Sixteen intact
word/span observations cover12texts and5sourcegroups;13 semantic regions are useful
local diagnostics, while coefficient/the/they examples remain whole-word-only where
neighbor ink crosses the proposed split. Preserve genuineff/fo/th joins rather than
forcing character separation. Existing39lines and174words overlap prior corpora and
must not be summed as independent new observations.

Concrete next tasks after product stability:

1. Reconcile reviewed whole-word/line labels, fullRGB crops, exact source transforms,
   document/session grouping and duplicate/exposure records in an immutable manifest.
2. Pre-register text-disjoint evaluation and explicitly state the frozen backbone's
   upstream source exposure. A new module split does not manufacture a new author-data
   holdout; require new unexposed sources for an independent author-style claim.
3. Replace the rejected independently-sampled-local residual assumption with a
   genuinely identifiable target: supervise a whole-word context/state representation
   using the observed complete word, jointly modeling morphology, geometry and ink.
   Keep frozen v21 as A and a shared-state B; a C arm must consume real continuous
   ink supervision, not repeated local generator calls.
4. Before training, implement the zero-state identity contract and nuisance controls
   separating word scale/slant/baseline from content/morphology/photometric differences.
   Do not use target-oracle alignment or true-condition reconstruction as generation.
5. Run bounded A/B/C only when source/task gates are met; test unseen words and full
   sentences with fixed seeds, missing-stroke/content gates, memory checks and author
   blind preference. Stop if only reconstruction/loss improves. Existing local-residual
   No-Go is not a ban on all future context models, but must not be renamed as success.

This audit starts no new network optimization, preference training or RSI loop.
