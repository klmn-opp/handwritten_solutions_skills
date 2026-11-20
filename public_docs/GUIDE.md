# CPU homework workflow

The runtime typesets checked answers; it does not independently solve an arbitrary
new problem. An external Agent must read the original and verify its answer.
The [Job contract](../references/jobs.md) and
[JSON schema](../references/job.schema.json) define the editable input.

Create `question.txt`, `answer.md` and `job.json` together. An example manifest:

```json
{
  "schema_version": 1,
  "generated_by": "Your Agent; answer independently checked",
  "profile": "bundled-default@v1",
  "realism": "baseline",
  "language": "en",
  "render": {"seed": 42, "dpi": 150},
  "problems": [{
    "number": "1",
    "statement": "Evaluate the integral of 2x from zero to three.",
    "sources": [{"path": "question.txt"}],
    "solution": "answer.md",
    "verification": {
      "status": "passed",
      "evidence": "Checked the antiderivative x squared and both limits.",
      "checks": [{"lhs": "3**2-0**2", "rhs": "9"}]
    }
  }]
}
```

The answer file can contain `The antiderivative is $x^2$, so the integral is $9$.`
Record source hashes when available. Text/PDF/image sources are copied into the
delivery; a reference to a page is not evidence that the Agent inspected it.

```sh
hwsol render --manifest /path/to/job.json --workspace /path/to/workspace
hwsol verify --job /path/to/workspace/runs/<run_id> --pixels
```

A successful run includes `rendered/homework_solutions.pdf`, page PNGs, source
copies, editable answers, a resolved manifest and reports. `--output` can export
a compatibility copy to a new directory. It must not overwrite existing work.
Verification checks identities, file hashes, counts and raster consistency.
Inspect mathematical meaning and readability separately.

Default pages are A4 white, 150 DPI; supported configuration permits up to 300 DPI.
Use plain, lined, grid, dots, or a local image/PDF background. Job-level paper can
be overridden per problem. The runtime can rewrap and reduce margins/font size
within its documented bounds; a solution that still cannot fit is rejected.
Do not remove required subparts or calculations to suppress a failure.

For a typesetting-only probe, use `hwsol render-probe --input <answer.md> --output
<new-directory>`. This legacy probe is not the complete one-problem-per-page Job
contract. See [rendering](RENDERING.md), [Profiles](PROFILES.md), and
[workspace behavior](WORKSPACE.md).

## Use a reviewed personal CPU Profile

The bundled font works immediately and needs no qualification evidence. A personal
ink Profile has a separate lifecycle so that importing handwriting does not change
your working default. Keep its ZIP and evidence on your own machine.

```sh
hwsol profiles import --source /path/to/personal-ink.zip
hwsol profiles list
```

An archive containing a personal ink recipe is imported as an inactive candidate,
even if another machine previously published it. Its declared source crops and provenance
travel with the Profile; local publication authority does not. For review, select
`"backend": "personal-cpu-experimental"` and the returned pinned `profile@version`
in a Job. Keep `"realism": "baseline"` and omit custom `render` overrides so the
Profile supplies its reviewed settings. Candidate output remains held and cannot
advance the workspace's delivery pointer.

After reviewing complete pages, the Agent or Profile maintainer assembles a local
evidence request. This is more than a successful render: it binds the exact Profile,
recipe, installed package and wheel, genuine Linux and macOS installation/output
receipts, independently checked mathematics, and the author's page review. Existing
installation reuse must reference its clean installation evidence. Qualification
also checks the current OS, architecture, Python minor version and actual text
layout capabilities. Matching package version numbers alone are insufficient:
Pillow builds may provide different BASIC/RAQM layout engines on different systems.

Given that reviewed evidence bundle, run:

```sh
hwsol profiles qualify-ink my-handwriting@v2 --evidence /path/to/review/request.json
```

The command validates the evidence and saves an immutable local qualification. It
returns a `qualification_id`; it does not publish, activate, upload or train anything.
Missing, changed or incompatible evidence is rejected. Ask the Agent to prepare or
update the review rather than changing an approval flag to bypass the check.

When you explicitly decide to make this qualified version available for ordinary
local rendering, use the returned identifier:

```sh
hwsol profiles publish-ink my-handwriting@v2 --qualification-id <qualification_id>
```

Here **publish** means publishing a version in your local Profile store. It performs
no Internet upload and does not change the active default. You can now select that
version explicitly in an ordinary Job with `"backend": "font-cpu"`,
`"profile": "my-handwriting@v2"` and `"realism": "baseline"`. Omit the Job's custom
`render` block: qualified ink requires the Profile's reviewed render settings.
Continue to check and verify each new answer and output; qualification does not
promise that every possible formula is supported.

Changing the default is a separate, explicit decision:

```sh
hwsol profiles activate my-handwriting@v2
# Return to an existing eligible historical version without deleting v2:
hwsol profiles rollback my-handwriting@v1
# Or return to the installed bundled font:
hwsol profiles activate bundled-default@v1
```

A historical personal ink version must also have valid local qualification for the
current build and environment before activation or rollback. A runtime upgrade,
changed Profile or different rendering capabilities can require renewed evidence;
the system rejects stale qualification instead of silently accepting it. Exporting
and importing a Profile does not transfer that qualification. Retain the old assets
and evidence so that a known working version remains reproducible.

## Preparing a local release candidate

Maintainers can build a local prerelease from an already committed source whose
`pyproject.toml` and project entry in `uv.lock` declare the same `X.Y.ZrcN` version
(N starts at 1):

```sh
python scripts/package_release.py --ref <frozen-commit> --version 1.0.0rc2 --output <new-directory>
```

This command resolves the ref once, reads and archives only that Git commit, and
uses the same public-file, private-asset and hash checks as stable packaging.
Uncommitted metadata or documentation cannot repair the selected commit. It
creates local wheel, sdist and Skill artifacts with `release.json` and
`SHA256SUMS`; it creates no tag, upload, Profile activation or production release.
Stable tagged builds still use `scripts/package_release.py vX.Y.Z --output DIR`.
The example RC number is not an issued release or an acceptance claim. Test the
exact new artifacts on the supported platforms and record their hashes before
claiming release readiness; an older package's installation evidence is not the
new package's verification.
