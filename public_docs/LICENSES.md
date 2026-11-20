# Distribution licenses and privacy

Project source is MIT, with the full [license](../LICENSE). The sole bundled
font is unmodified Patrick Hand under SIL OFL 1.1. Its license and provenance are
included in [the font asset directory](../src/hwsol/assets/presets/patrick-hand).
The package metadata declares MIT AND OFL-1.1. The bundled font is an engineering
default, not a personal handwriting sample or model.

Runtime dependencies are pinned in `pyproject.toml` and `uv.lock`, including the
specific Handnote source revision. Their own licenses apply. Matplotlib's math
fallback fonts and PDFium/native components carry separate notices in their
installed distributions. A source license does not establish rights to model
weights, handwriting datasets or user submissions.

Optional adapters do not ship external training corpora or neural checkpoints.
Users must obtain compatible assets with suitable permissions in their own
research environment. An upstream code repository's MIT license must not be
treated as authorization to redistribute its associated data or weights.

Public archives exclude original personal assignments, Profile stores, private
model weights, local research logs, operational handoffs and the user's master
instruction. Path and content scanning is an engineering release check, not a
complete legal audit or a guarantee about unrelated user-added files. Review
permissions and platform-specific notices before distribution.
