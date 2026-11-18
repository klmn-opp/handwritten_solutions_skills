# Contributing

Use Python 3.12 or 3.13, Git and uv. From the checkout run the following commands before proposing a change.

```sh
uv sync --locked
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -ra
uv build --out-dir dist
uv run --no-sync python scripts/smoke_wheel.py --dist dist
```

Keep original problem numbers, source Markdown, reference hashes and failure reports when changing layout or verification. For changes to fonts or neural experiments, record the upstream revision and the licenses for code, fonts, weights and data separately. Use clearly marked synthetic input in tests. Keep personal samples, assignments, environment files and model weights in a private data directory.

Include a small reproduction when reporting an issue. Use redistributable synthetic material and retain the relevant error message, platform, Python version and Profile version. Changes to reproducible builds should update the lock, documentation and installation checks together.

Maintainers can generate local release files from an annotated version tag. Git bundles contain the full history reachable from that tag. Review the history and [licensing inventory](docs/LICENSES.md) before sharing a release. The packaging command stages the tag in a temporary directory, builds the source and wheel, packages the Skill and writes SHA256SUMS.

```sh
uv run --no-sync python scripts/package_release.py v0.3.0 --output dist/v0.3.0
uv run --no-sync python scripts/smoke_wheel.py --dist dist/v0.3.0
```

The optional MarkdownToHandwrite tests use HWSOL_MTH_PYTHON to locate an independent compatible interpreter. [Verification records](docs/VERIFICATION.md) describe the tested configurations.
