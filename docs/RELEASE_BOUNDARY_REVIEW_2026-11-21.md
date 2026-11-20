# First public release: independent boundary review

Review date: 2026-11-21. This is a local engineering review, not a public
distribution document or an assertion that publication has occurred.

Reviewed development HEAD: `507651999ddcb6d2be523c41b2110c2dcc658777`, on
`release/v1.0.0`. The parent release task was concurrently editing version/build
files. The observations below describe the inspected implementation and historical
evidence; final artifact hashes and final qualification belong in the release run.

## Decision

**Conditional Go for preparing a public CPU release; No-Go for pushing the current
development repository's history or treating RC2 receipts as 1.0.0 receipts.**

The user has requested GitHub upload, packaging, documentation and conservative
cleanup. The safe publication unit is the existing audited public distribution
snapshot, with a new public Git history. It is not this private development
history. Creating an export checkout does not replace or rebuild the development
repository. The private development repository, original handwriting and all
historical evidence remain local.

The new stable package still needs its own completed installation/rendering
receipts. The previous release-candidate evidence is useful regression evidence,
but cannot establish an exact-build qualification for a different package.

## Public and private content

`scripts/package_release.py`, `public_build_backend.py`, and
`public_docs/distribution.json` implement a useful existing boundary:

- Explicit runtime, test, example and documentation allowlists; public templates
  replace local README, Skill and selected operational documents.
- No source scans, workspace runs, private Profile stores, checkpoints, research
  tree, master instruction, execution handoff, SSH files or local credentials.
- The sole bundled font is the unmodified licensed Patrick Hand font. It must not
  be described as the author's personal handwriting or a learned personal model.
- Per-file SHA inventories and archive audits check the staged bytes and detect
  unexpected entries, broken local Markdown links and selected private-path text.

This boundary is stronger than relying on `.gitignore`. A reachable-history
inspection found 2,542 object/path entries, including 836 research and 767 docs
entries. These are object/path entries, not independent documents. No paths under
`workspace`, `source_pdf`, `profiles`, `samples`, `checkpoints`, `.agents`, `.aws`
or `.codex`, and no `.pt`, `.pth`, `.ckpt` or `.safetensors` path suffixes, were
found in that inventory. This does **not** make the history public-safe.

An additional scan examined 1,377 unique textual blobs from that history. Matching
object/path entries included 190 private run references, 132 original-scan
references and 33 absolute home-directory references. Examples of affected file
names include `docs/MASTER_EXECUTION_PLAN.md`, `docs/CURRENT_STATE_AUDIT.md`,
`docs/PERSONAL_HOMEWORK_CORPUS.md` and `PROJECT_AUDIT_2026-11-20.md`. The scan
reported categories and filenames, not private text. Removing such documents at
the tip of an existing branch would not remove their ancestors.

Minimum safe publication procedure:

1. Freeze and retain the private development source commit locally.
2. Build/stage through the existing explicit public allowlist; audit the completed
   source, wheel, sdist and Skill archives.
3. Review final demo assets and their embedded metadata. Use a newly rendered
   bundled-font demo with synthetic checked questions; do not publish a private
   Profile or author-ink demo merely because source publication was requested.
4. Initialize a separate public export checkout from only the approved files.
   Do not copy `.git`, push existing refs/tags, or upload a private Git bundle.
5. Publish only the reviewed export and release assets to the user's identified
   repository. Preserve both source identities in the release record.

The current text scan is not a comprehensive secret scanner. In particular, its
private-path content scan targets document formats and does not inspect all Python
strings or PDF/PNG metadata. Final public bytes therefore need a supplementary
scan/review; no result here claims a guarantee about arbitrary future files.

## Build identity and qualification

The previously qualified runtime was development commit
`3f8bff9c1659ecb9636f45764d103a0bccbc4911`, version `1.0.0rc2`. Its wheel SHA was
`3e6a07279b78bd9f0bffc80c1c579c9e393c2ce248209583351a8b30d22c3c08`.
Real clean Linux and macOS installation/output evidence exists in the handoff;
the prior tested environments include Linux Python 3.12/3.13 and macOS 15.6.1
arm64 Python 3.12/3.13. This is actual RC2 evidence, not a completed 1.0.0 claim.

Qualification binds package/version, code/dependency identity, Profile/recipe,
actual wheel, installation/output receipts, platform/Python and Pillow text
capabilities, mathematics review and author approval. Updating metadata to
`1.0.0` changes the wheel and runtime identity. Reusing or relabeling the RC2
qualification would defeat that protection. Rebuild the stable wheel and rerun
the applicable actual environment checks; generate a new qualification if the
private personal CPU candidate is to be declared qualified on the new build.
The bundled default CPU font does not require a private personal qualification.

`_build.json` embeds the development source commit used by the package builder.
The public export Git commit will necessarily differ. `build_identity()` uses
the checkout's Git HEAD when present, otherwise the embedded build identity.
Consequently a public source checkout and an installed frozen wheel are distinct
identity contexts. Preserve an explicit mapping of:

- Private frozen development commit and selected version/tag.
- Public export Git commit and public source inventory SHA.
- Wheel, sdist, Skill archive and checksum-manifest hashes.
- The exact installed artifacts used in each acceptance run.

Do not push private ancestry to make the hashes match, or falsely label the public
export commit as the development commit. A qualification must refer to the
identity it actually tested.

The stable-tag source resolver checks the pyproject version against the tag. At
inspection time its lockfile self-version check was narrower, applying to the
local-prerelease path. The stable build must explicitly verify that `uv.lock` and
`pyproject.toml` both identify `1.0.0`, and that locked installation succeeds.
Build real stable archives; do not rename RC2 filenames or edit receipts.

## Installation and advertised scope

The public PEP 517 backend verifies the entire staged source. A `.git` directory
inside a public Git checkout is not an allowed distribution entry, so directly
running `pip install .` or `uv build` in that checkout will currently be refused.
The supported editable `uv sync --locked` path bypasses distribution staging;
ordinary archives should be built from `scripts/package_release.py --stage` and
installed from the resulting wheel/sdist. README instructions must reflect the
actual tested path. Do not advertise a direct Git-root wheel build unless that
behavior is deliberately corrected and verified.

The wheel supplies the CPU CLI/runtime. The Skill/source archive supplies Skill
instructions, examples and documentation; installing a wheel does not by itself
register an Agent Skill with every host application. The host Agent reads the
problem, solves it and independently verifies the answer. The renderer typesets
the checked Job and performs engineering validation; it is not an autonomous
mathematical solver or a proof checker.

The runtime does not require Torch/GPU and does not ship trained personal neural
weights. Initial dependency installation includes the pinned Handnote Git source
and external binary dependencies, and may need network access/Git; the wheel alone
is not a complete offline dependency bundle. The public default font and fallback
math fonts are not equivalent to the private author's fully reviewed Profile.
Explicit unsupported-structure rejection and visible fallback provenance should
remain documented.

The CI matrix includes Windows, but configured CI is not evidence of a completed
Windows validation. Advertise only actually executed platform/version coverage.
At this review the targeted format check passed for qualification, identity,
packaging and their related tests; that check does not substitute for package
installation, complete Job rendering or author style review.

## Licenses and prior release wording

The source is MIT. The package metadata declares `MIT AND OFL-1.1` and includes
the project license plus Patrick Hand's OFL notice. All three original bundled
font/provenance file hashes matched the recorded pinned upstream bytes during
this review. Preserve the OFL notice and font provenance in the archives.

Dependencies retain their respective licenses and installed notices, including
Matplotlib fonts and PDFium/native components. `scripts/license_inventory.py`
can snapshot installed package metadata and license-file hashes. Code licenses do
not establish permission to redistribute handwriting scans, private Profiles,
external training corpora or model weights. This is a distribution inventory,
not legal advice or an assertion that metadata alone resolves all permissions.

Local tags `v0.3.0`, `v0.4.0` and `v0.6.0` exist. No Git remote was configured at
review time. Historical documents using “released” are not independent proof of
an existing GitHub release; equally, absent local remotes do not prove no prior
upload ever occurred. Describe those as historical local tags/builds unless
actual remote publication evidence is subsequently obtained. Record the new
GitHub URL and release state only after the operation really succeeds.

## Review methods and remaining ownership

Read-only methods used: `git rev-parse`, branch/tag/remote inspection,
`git rev-list --objects --all`, selective `git cat-file blob` category scans,
inspection of the public builder/backend/allowlist/identity/Skill/license files,
font-provenance SHA checks and targeted `ruff format --check`.

The parent release task owns final package creation, artifact inspection,
installation and macOS/Linux qualification, credentials/repository selection,
public export, GitHub upload, cleanup and final audit. This report author changed
no runtime, historical asset, release pointer, remote, archive or qualification.
The only authored file is this local report.
