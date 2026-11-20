# M2.1 — actual platform acceptance

`cross_platform_candidate_check.py` is a standalone standard-library coordinator.
It runs the installed `hwsol` CLI with a fresh isolated Profile store and workspace.
It does not import research modules, upload private files, select a better seed,
change the production default, or pretend that Linux is macOS.

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

## Current status

The author has offered a real Mac and authorized copying the selected private
Profile/Job to that owned machine. A working SSH endpoint is now reachable;
authorization of a temporary login key is pending. No Mac execution has occurred.
This scoped authorization does not extend to third-party Mac hosting.
macOS acceptance remains **not executed / No-Go**, regardless of Linux smoke results.
The local smoke uses the previously installed8da224f4 Linux wheel and previously
reviewed Jobs, so it establishes harness behavior only, not unseen-content or new
clean-install evidence. Final immutable run:
`20261120T154837649227Z-portable-platform-final-harness-4b3eb719`.
Its `reports/acceptance/receipt.json` passes actual Linux candidate→previous→candidate
rendering; `reports/darwin-mismatch/receipt.json` fails the OS gate before any CLI
command. All four retained canonical Jobs pass pixel verification. Earlier full
six-page smoke4221a739 passes; first fixture failure8c4c04dd is retained.
When the Mac connection is supplied, use the new end-to-end Job, not those old fixtures.

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
Use that exact artifact and new Job for the forthcoming Mac run. The initial
build helper attempt004eec3b is retained: writing the Skill archive after wheel
build included generated bytecode and was correctly rejected. Fresh43bb9000 writes
the Skill archive from the pristine stage first; the runtime source is unchanged.
