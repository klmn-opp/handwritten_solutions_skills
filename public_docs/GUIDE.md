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

## Preparing a local release candidate

Maintainers can build a local prerelease from an already committed source whose
`pyproject.toml` and project entry in `uv.lock` declare the same `X.Y.ZrcN` version
(N starts at 1):

```sh
python scripts/package_release.py --ref <frozen-commit> --version 1.0.0rc1 --output <new-directory>
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
