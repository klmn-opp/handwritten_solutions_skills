# Contributing to the CPU distribution

Use Python 3.12 or 3.13 and the locked uv environment. Keep runtime code independent
of Torch and private assets. Add redistributable, synthetic reproductions for
content, layout and source-reporting defects. Preserve problem numbers, source
hashes and failure reports. Check the complete page as well as implementation
tests; tests cannot establish personal visual quality.

```sh
uv sync --locked
uv run --no-sync python -m pytest -ra
uv run --no-sync ruff check src scripts tests
python scripts/package_release.py --stage .release-work/public-source
cd .release-work/public-source
uv build --out-dir ../../dist
```

Return to the versioned checkout before invoking tag-based release preparation.
Public packages are prepared with the versioned allowlist in
`public_docs/distribution.json`. Only the staged public source uses the public
README, Skill and guides; local research history is unchanged. The release tool
requires its selected Git tag to contain that public layer and to match the
Python version. It refuses older refs without the layer rather than replacing
their historical contents from the current working tree.

```sh
python scripts/package_release.py vX.Y.Z --output dist/new-release-directory
```

The PEP 517 backend refuses normal wheel/sdist builds and metadata creation from
an unprepared private tree. Editable `uv sync --locked` remains available for
local development. An extracted public sdist retains its receipt and can rebuild.

This command creates local reviewable artifacts, not a remote publication.
Unreviewed Git bundles and full history are not distribution artifacts. A public
release still requires explicit maintainer/user approval and all applicable
platform, content, privacy and licensing checks. See [licenses](public_docs/LICENSES.md).
