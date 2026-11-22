# Product priorities

## Delivered in 1.2.x

The author's accepted seven-exercise model is now the default backend for the
bundled author Profile on white paper. It keeps CPU mathematical layout, ships only
the audited inference arrays and glyph subset, runs without Torch, and falls back
per glyph. See [the model page](NEURAL_MODEL.md).

## This generation: personal handwriting by default

A clean install must render with the author's latest accepted personal CPU Profile,
including the final word-gap adjustment. Existing explicit user defaults stay in
place; Patrick Hand remains an optional compatibility preset. The release ships
only the authorized runtime derivative: fonts, compiled contours, the audited
`nextgen-v1` inference arrays/glyphs and necessary configuration, without original
scans, private paths, optimizer state or unreleased research checkpoints.

A new user must have a short, supported path to create their own Profile:
print the character worksheet, write and scan it, review the actual labels, build
and preview with one command, then activate the accepted version. Repeated imports
create cumulative versions and preserve originals. See [the practical guide](PROFILES.md).
CPU font extraction/calibration is the supported method in this generation;
continuous neural personalization is separate research.

Release acceptance includes a clean installation, omission of `profile` in a real
Job selecting the author preset, complete PDF/PNG generation, provenance and
page-boundary checks, Profile creation and rollback, and package privacy/license
audits. A new artifact must have its own platform evidence.

## Next: flexible text and mathematics

Inline mathematics already exists. The next composer should choose inline or
separate display placement according to the sentence, formula size and page space,
so explanations and calculations are not forced into alternating text/formula
rows. It must preserve mathematical semantics, subproblem order and stacked
fractions, including exponent fractions; units such as m/s stay intact.

Evaluate paragraph wrapping, short formula chains, tall fractions, matrices and
dense derivations against the accepted personal baseline. Reject clipping and
missing content. Do not describe existing inline support as a completed adaptive
layout planner.

## Next: clear background selection

The Job schema already offers `plain`, `lined`, `grid`, `dots`, `image`, and `pdf`,
with per-problem overrides. Make these easy to select and preview, and test the
personal renderer on each. Preserve its accepted visible word gaps; ruled-paper
line snapping must not disable the semantic layout required by those gaps.
Background contrast, margins and legibility require actual page checks.

## Experiment: write in the source PDF's blank areas

Start with explicit page/rectangle answer regions, then add proposed whitespace
regions for review. Preserve the original PDF; write a new output with an answer
layer. Detect intersections with printed questions, diagrams and existing ink,
respect page rotation/crop boxes, and keep answer content complete. If space is
insufficient, report it and offer continuation pages rather than shrinking away
necessary steps or covering the question.

Using a PDF as the current full-page background does not yet perform these tasks:
it rasterizes a selected page and places text using ordinary page margins. Region
placement, occupancy detection and original vector/text preservation need their
own implementation and tests. This feature remains experimental until validated
on both text PDFs and scanned worksheets.
