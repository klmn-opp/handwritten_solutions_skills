# Distribution licenses and privacy

Project source and the author's authorized compiled runtime outlines/configuration
are MIT, with the full [license](../LICENSE). The default `author-default@v1`
is the author's accepted personal CPU style. It contains reusable fonts and
compiled contours, with the owner's explicit permission to distribute them as
this open-source project's default. Raw scans, private source paths, full training
corpora, optimizer state and other research checkpoints are not included.

The `nextgen-v1` runtime files contain the two frozen endpoint checkpoints as
value-identical float32 arrays plus the 1,706 inference glyphs needed by the default
backend. The owner authorized this minimal derivative. It is released under the
project's MIT terms; the original `.pt` files, absolute paths and optimizer state
stay local.

The personal font files derived from Patrick Hand retain SIL OFL 1.1 and their
original embedded copyright/license notices. The unmodified Patrick Hand remains
an optional `bundled-default@v1` preset. Its [license and provenance](../src/hwsol/assets/presets/patrick-hand)
remain included. The package metadata declares MIT AND OFL-1.1; the MIT source
license does not replace the OFL terms on those fonts.

Runtime dependencies are pinned in `pyproject.toml` and `uv.lock`, including the
specific Handnote source revision. Their own licenses apply. Matplotlib's math
fallback fonts and PDFium/native components carry separate notices in their
installed distributions. A source license does not establish rights to model
weights, handwriting datasets or user submissions.

Optional adapters do not ship external training corpora or neural checkpoints.
Users must obtain compatible assets with suitable permissions in their own
research environment. An upstream code repository's MIT license must not be
treated as authorization to redistribute its associated data or weights.

Public archives exclude original personal assignments, Profile stores, unreleased
research model weights, local research logs, operational handoffs and the user's
master instruction. Path and content scanning is an engineering release check, not a
complete legal audit or a guarantee about unrelated user-added files. Review
permissions and platform-specific notices before distribution.
