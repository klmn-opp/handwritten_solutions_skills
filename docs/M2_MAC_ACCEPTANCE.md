# M2.1 — actual platform acceptance

`cross_platform_candidate_check.py` is a standalone standard-library coordinator.
It runs the installed `hwsol` CLI with a fresh isolated Profile store and workspace.
It does not import research modules, upload private files, select a better seed,
change the production default, or pretend that Linux is macOS.

The latest completed evidence is [RC2 native qualification](#rc2-native-installation-and-qualified-cpu-lifecycle):
actual Mac3.12/3.13 fresh installs and the ordinary qualified CPU workflow pass in
isolated stores. Earlier4a/RC1 results below remain distinct historical evidence.

## Run on the actual Mac

Prerequisites: a real macOS machine, Python3.12 or3.13, Git, and the explicitly
selected local wheel, candidate Profile ZIP, previous Profile ZIP and complete
new Job input folder. Keep the private Profile and homework on the user's machine.
Creating an SSH host or uploading these private files to a third-party Mac service
requires separate explicit authorization; this document grants none.

The current wheel's Handnote dependency is a pinned public Git commit. A fresh
installation downloads public dependencies, so it needs network access and Git.
A universal wheel filename alone does not prove macOS installation works. This
checker deliberately does not claim offline installation from an incomplete cache.

```sh
python3.12 cross_platform_candidate_check.py \
  --require-system Darwin \
  --wheel /path/to/selected.whl \
  --profile /path/to/candidate-v2.zip \
  --rollback-profile /path/to/candidate-v1.zip \
  --job /path/to/new-homework/job.json \
  --output /path/to/new-macos-acceptance
```

The output directory must not exist. A failed attempt retains its logs and must
use a new output path for a retry. Default mode creates a new virtual environment
and installs the supplied wheel. `--runtime-python /path/to/installed/python`
audits an existing installation instead, explicitly recording that it is **not**
a new clean installation. This mode is useful for diagnosing an already installed
runtime but cannot alone satisfy M2.1's clean-install gate.

## Evidence and interpretation

`receipt.json` records actual OS/architecture, launcher/runtime Python, package
version, source commit, input wheel/Profile/Job SHA, exact installed wheel members,
every subprocess command/status/duration and the actual outputs. The check does:

1. Reject an OS mismatch before rendering (`--require-system Darwin` on Linux fails).
2. Verify installed package files match the supplied wheel and no research/Torch is
   importable in the target interpreter.
3. Render and pixel-verify the ordinary bundled default control.
4. Import both private versions as inactive candidates.
5. Render/verify the caller's complete Job using candidate → previous → candidate.
   Profile pins change; original problem/solution bytes do not.
6. Check PDFs and per-problem PNG inventory; the repeated candidate PNG hashes match.
7. Export the candidate, reimport that archive into a second fresh local store,
   export again, and compare every member to the original; neither import activates it.
8. Verify default/settings/latest remain exactly as after the bundled control.

The checker is not a solver or a mathematical proof system. Supply the newly
Agent-read, solved and independently checked Job prepared by the end-to-end stage;
keep that stage's source/transcription/solution evidence. Replaying old pages only
verifies the harness and never counts as new-content acceptance. Pixel verification
checks the output's own PDF/PNG consistency; cross-platform reference differences
and author visual approval remain separate review steps.

Do not upload the whole result folder publicly: it includes private Profile data,
original task material, paths and generated handwriting. The receipt and selected
redacted diagnostics can be returned to the local Agent by the author's chosen means.

## Initial native4a acceptance

The owner has authorized and authenticated the private connection. Actual native
acceptance completed on Darwin arm64, macOS15.6.1 (24G90), in immutable run
`20261120T161118596530Z-m21-native-macos-bdb2998d`. Python3.12.16 and3.13.7 each
created a genuinely fresh environment and passed all eight checker operations,
including the new4 Job, profile export/reimport and v3→v2→v3 rollback. All88 installed
wheel members match the exact Linux-tested4a artifact. No global installation,
production/default activation, private cloud upload, or credentials in Git occurred.
The isolated3.12 toolchain was downloaded beneath the new acceptance directory.

The runSHA is `2c7586d94e767fd89b2c6a5132a45a47b6a8080c682ff89f46d2db8f1bf702a6`.
Both portable receipts truthfully state `clean_install_completed=true`. Their SHA:
3.12 `05409d7145c3ef5cdb5c00f98feb47a858004cdfb8d18284949753f45545a4a8`;
3.13 `e90f122eb98ac27e9d361fbeda35db0340d3ffe262b893ea14923350e31b30da`.
Native PDFs/PNGs and full logs are in `rendered/native/python-3.12` and `python-3.13`.

The two Mac outputs are mutually byte/pixel/source/geometry exact. They differ from
Linux:1.19–2.23% page pixels, mainly horizontal placement, seven f calibration
admission decisions, one title d variant and seven matrix-digit translations of2–7px.
There is no new mathematical semantic change, external-font fallback, or confirmed
boundary defect. Four complete pages and changed letter/matrix details were reviewed.
These are actual output differences, not merely PNG compression.

Comparison run `20261120T161909117292Z-m21-native-platform-comparison-b71b610e` records
all pages, per-item geometry/source differences, aligned crops and visual comparisons.
Measured cause: both use Pillow13.0.0/FreeType2.14.3, but Linux's binary includes RAQM
and Mac's does not. Handnote selects Pillow's default layout engine: RAQM on Linux,
BASIC on Mac. The same font's explicit BASIC metrics match across OS; default metrics
do not. Existing clearance gates correctly retain original outlines where needed.
No kernel or gate was modified to force equality. Native functional compatibility is
Go for this tested4a/v3 scope; cross-OS pixel identity is not met. New RC packages,
word-gap changes and author style admission require separately bound evidence.

The local smoke uses the previously installed8da224f4 Linux wheel and previously
reviewed Jobs, so it establishes harness behavior only, not unseen-content or new
clean-install evidence. Final immutable run:
`20261120T154837649227Z-portable-platform-final-harness-4b3eb719`.
Its `reports/acceptance/receipt.json` passes actual Linux candidate→previous→candidate
rendering; `reports/darwin-mismatch/receipt.json` fails the OS gate before any CLI
command. All four retained canonical Jobs pass pixel verification. Earlier full
six-page smoke4221a739 passes; first fixture failure8c4c04dd is retained.
The completed native run uses the new end-to-end Job, not those old fixtures.

The final export/reimport addition is exercised in immutable run
`20261120T155532252586Z-portable-platform-reimport-smoke-a33b4f61`; all asset hashes
and inactive admission remain exact. This is Linux existing-install evidence only.


## New four-problem clean Linux installation

Exact source `4a0e52ac1f8c643b3a3714c0867cbea562fbee6b` is independently installed in
fresh Linux Python3.12.16 and3.13.18 environments in immutable run
`20261120T155718248776Z-m21-clean-installed-new-job-43bb9000`.
RunSHA:`82d6a7dbfb4c957279f7451252c08d36379d661683d7f585ba7da792ae8696ac`.
The new four-problem Job (ab6496e5) is rendered with privatev3, rolled back tov2,
and reselected tov3. Both Python outputs match the accepted four-page reference
and each other byte/pixel/semantic/source/geometry/formula-wise. The default control,
all eight acceptance operations, and public archive privacy audit pass.

`reports/installations.json` records actual fresh dependency installation from the
locked offline cache and exact wheel installation, linking portable receipts by SHA.
The portable subreceipts correctly retain `clean_install_completed=false` because
the separate installer, not the checker itself, created those environments. Both
pieces of evidence must be assessed together; do not rewrite the subordinate receipt.

The public wheel is in that run's `reports/dist/`:
`handwritten_solutions-0.6.0-py3-none-any.whl`, SHA
`df96bf879070431c85c8e8b6541235afc14162098b5375ab98c97a1db704accb`.
It is an unreleased source-verification build, not a newly published0.6.0 package.
That exact artifact and new Job were used for native run bdb2998d. The initial
build helper attempt004eec3b is retained: writing the Skill archive after wheel
build included generated bytecode and was correctly rejected. Fresh43bb9000 writes
the Skill archive from the pristine stage first; the runtime source is unchanged.

## Canonical RC1 package and final word-gap Profile

The actual canonical `1.0.0rc1` wheel from source
`884f397383b8af204248d2cd3aa2570f0eab74ad` is separately installed on the real Mac in
`20261120T162628724474Z-m21-native-macos-ce6bbb69`. Both Python3.12.16 and3.13.7 pass
all operations with `clean_install_completed=true`. WheelSHA:
`44018419a7652b1a3fe7bc0c6d607eeff1405d3b840543b6a29952abf205a900`.
RunSHA:`4a4c44cb93a6df81cf8351499521cb19ec7ab0a955e55b0e676fb926606b09fa`.
Only already downloaded public dependency cache entries were reused when the network
slowed down; no installed package directory, private store or output cache was copied.
All cache additions and their hashes are recorded; existing files were not overwritten.

The author's subsequent +8% word-gap request produces v5 (1.06×1.08=1.1448).
`20261120T163915577202Z-m21-native-macos-67021e67` checks v5→v4→v5 in new stores and
outputs using those exact validated RC1 interpreters. It links the genuine clean
parent and correctly reports `clean_install_completed=false` for this Profile-only
increment. RunSHA:`598da60982c971cbe89ff294637f661e71ea105d126f1defc717e9b6142c0183`.
Both native versions pass; the resulting four pages match each other exactly.

Final comparison/review:
`20261120T164142331385Z-m22-final-native-profile-review-b95703b8`.
Nativev5 preserves all formula groups,2594 visible semantic items and page/line
structure from nativev4. Two f calibration decisions and eight±1px y placements
change; those details were inspected, not hidden by claiming translation-only
changes. Four complete pages and both changed f regions have no newly confirmed
math, missing-content, contact or boundary defect. Cross-Linux pixel differences
remain explained by the measured RAQM/BASIC capability difference. Source selections
are therefore not claimed identical across platforms. The author waived another
word-gap review; this does not waive later runtime/package qualification gates.
No production default or published release changed.


## RC2 native installation and qualified CPU lifecycle

Exact runtime/source: `3f8bff9c1659ecb9636f45764d103a0bccbc4911`, package `1.0.0rc2`.
The canonical wheel SHA is
`3e6a07279b78bd9f0bffc80c1c579c9e393c2ce248209583351a8b30d22c3c08`.
It was independently clean-installed on Linux in2880993d and on the actual owner Mac
in `20261120T165923744719Z-m21-native-macos-a8b00dcb`.
Native runSHA:`d025e7db5dfe32143a40fbe6bba25da03b2e6da9612145df7bc280bd6b674d0a`.
Both native3.12.16/3.13.7 installation receipts report genuine clean installs and
all eight acceptance operations pass. Public dependency cache reuse is explicitly
recorded; each runtime/store is new. Native text capabilities are captured in the
actual portable receipt, including BASIC versus LinuxRAQM.

The actual qualification bundle is
`20261120T170521217308Z-m23-local-qualification-bundle-bc7b1c05/solutions/qualification/request.json`,
SHA`87f6e847d7f9be6fd16a97ed25ddf1241727efcc07208bdc2b10b90ebcf56021`.
Its254 hashed files bind all four environments, exact wheel/build/Profile, complete
PDF/PNG Jobs, clean-install evidence, historical author statements and independently
checked mathematics. NewRC2 v5 outputs and native v4 rollback pages must exactly
match the genuinely reviewed corresponding RC1 pages before the assembler writes
any approval record. Existing Linuxd and Macf changes are explicitly inventoried.
The author record means acceptedv3 style plus the prescribed spacing waiver; it
does not claim that the author freshly inspected RC2 or every native Mac page.

The qualification workflow itself was then executed on both native interpreters in
`20261120T170809913530Z-m23-native-qualified-lifecycle-fd78ef1d`.
RunSHA:`d419de16dd0e93a05b9191bef3e8cbaf5311c9fc7f5796cbfdcf473e48585652`.
These lifecycle checks reuse the exact clean-installed RC2 interpreters and correctly
report **no new package installation**. They use fresh temporary stores/workspaces.

Both actual native versions pass:

- Import starts as a candidate; activation before publication is rejected.
- `qualify-ink` accepts the exact evidence without publishing or activating.
- `publish-ink` creates a local published version while leaving the default unchanged.
- Ordinary `font-cpu` generates and verifies all four pages using qualified personal ink.
- Explicit activation in the test store produces identical pages; rollback restores the bundled default.
- Export/reimport preserves asset hashes, loses publication authority and cannot activate in the second store.

The local qualification ID is
`6c58cc6a45e714c578ff43cc8d006c7d98a2e151459906c9f6062d047f16a94a`.
Native lifecycle receiptSHA:3.12
`f7c89a43005b98e189a8bee94ee4d5b1f23d464c7bff4e5a693f366ecb149f37`;
3.13 `bdbe1324dcfa91182f5a35041dc7f51443e0c2da93e7654e0ef7621902fcf831`.
[Actual ordinary CPU Mac PDF](../workspace/runs/20261120T170809913530Z-m23-native-qualified-lifecycle-fd78ef1d/rendered/native/python-3.13/qualified/homework_solutions.pdf)
and its `pages/` PNGs are retained, alongside activated-default output and all logs.

Every qualified/default output is pixel/source/formula/geometry exact to its own
native candidate. Cross-Linux pixel identity is still not claimed. Functional
native compatibility and this isolated qualified CPU lifecycle are **Go for the
recorded RC2/Profilev5/four-Job scope**. This is not a global publication or activation:
production main/latest/default, historical data and the owner's existing Profile
store remain unchanged. No Torch/research import, new neural training or external
private-data hosting is part of these native runtime checks.
