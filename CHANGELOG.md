# Public distribution notes

## 1.2.1 — platform admission correction

Adds the two actual GitHub Linux text environments (x86_64, Python 3.12 and 3.13,
FriBidi 1.0.13) to the bundled author-preset admission, alongside the maintainer
Linux 3.12 environment (FriBidi 1.0.8). Unverified macOS/Windows text stacks remain
excluded. The public CI matrix is restricted to the platforms this release actually
verifies. No model weight, glyph, rendering algorithm or output claim changes from
1.2.0.

## 1.2.0 — next-generation default backend

Clean installations that use the bundled author Profile now render white-paper
Jobs with the frozen `nextgen-v1` model: personal CPU layout, bounded English and
mathematical glyph variation, deterministic ink/tone correction and integer word
spacing. The model runs on CPU with NumPy only; PyTorch is not required.

The release ships the two accepted checkpoints as value-identical float32 arrays
plus the minimal inference glyph subset. No optimizer state, raw scan, full
training corpus or other version's weights are included. Every array is checked
against its source checkpoint; publication does not retrain or modify any weight.
Unsupported symbols, structure elements and failed guards fall back to the stable
CPU glyph and record the reason. Non-plain paper delegates to the stable CPU path;
the bundled author preset currently rejects ruled/grid paper because its final word
gap requires semantic layout.

Actual installation and rendering evidence for this version covers Linux x86_64
with Python 3.12; macOS and Windows are not re-verified in 1.2.0. Explicit
`"backend": "font-cpu"` and the Patrick Hand example remain available.

## 1.1.0rc1 — personal-first release candidate

Clean installations select the author’s latest accepted personal handwriting,
including its final word-gap setting. The shipped runtime derivative removes raw
crops and private paths while retaining exact outlines and declared font licenses.
Explicit user defaults and the optional Patrick Hand example remain available.

`profile build` provides a reviewed-template scan → CPU font → preview workflow;
activation remains explicit. This is font construction, not neural training.
Flexible paragraph/math layout and source-PDF blank-area filling are planned
separately; see [the roadmap](public_docs/ROADMAP.md).

## 1.0.0 — CPU release

A complete local Agent Skill and CPU workflow: structured homework Jobs, one major
problem per page, deterministic mathematical layout, PDF/PNG and editable answers,
explicit provenance/fallback reports, portable Profiles and separate qualification,
local publication, activation and rollback. The package includes a reproducible
licensed-font demo. Runtime rendering does not require GPU or PyTorch.

The host Agent supplies problem understanding, solutions and independent checks.
Neural personal handwriting research remains separate; no private scans, personal
fonts, training data or model weights are distributed. Tested platform evidence and
remaining mathematical limits are recorded in the release documentation.

## 1.0.0rc2 — local qualification candidate under acceptance

Adds separate local evidence qualification, ink publication and explicit activation.
Qualified personal CPU Profiles can use the ordinary font-cpu Job path. Qualification
binds the exact Profile, recipe, package and actual platform text capabilities to
complete homework and author/math review evidence; imports do not carry authority.
The default bundled CPU workflow is preserved. No private Profile or neural weight
is distributed. This source change needs its own installation/acceptance receipts;
previous candidate results do not establish acceptance of this version.

## 1.0.0rc1 — local prerelease under acceptance

The default remains the CPU font backend. The optional private personal CPU recipe
uses explicit candidate selection, original glyph provenance and strict geometry
gates; it remains held and cannot activate a production default. This prerelease
adds visible word-gap scaling for semantic layouts, rejects refinements that create
new ink contacts, and reports the installed application/build identity in `doctor`.
It does not include private Profiles, neural weights or an approved neural backend.

This version identifies a testable local package; it is not a declaration of completed
cross-platform acceptance or production qualification. Consult its exact build and
acceptance receipts before making a release claim.

The exact package version is declared in `pyproject.toml`; the build receipt
records source commit, dependency lock and file hashes. This public-documentation
layer does not change that version or declare a new release.

The CPU runtime provides Markdown/LaTeX homework layout, one top-level problem
per page, source and verification records, PDF/PNG output, immutable workspaces,
bundled fonts, personal Profile versioning and the CPU Studio. Optional experiment
adapters remain separate. Private research experiments and author-specific
quality claims are intentionally absent from public archives.
