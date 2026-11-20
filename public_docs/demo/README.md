# Actual public CPU demo

These two original problems were written, solved and independently checked for
this demonstration. They contain no historical homework or private handwriting.
The checked inputs are in [examples/release-demo](../../examples/release-demo/).

The PDF and 150 DPI page images are unmodified renderer output. Patrick Hand is
the licensed bundled default (OFL-1.1); MathText uses STIXGeneral for the radical
and prime reported in [provenance.json](provenance.json). This is not a zero-fallback
or neural-generation demonstration. Third-party licensing is documented in
[LICENSES.md](../LICENSES.md).

Reproduce after installation, from the source package root:

```sh
hwsol render --manifest examples/release-demo/job.json --workspace workspace
hwsol verify --job workspace/runs/<returned-run-id> --pixels
```

The original rendering used the verified 1.0.0rc2 runtime with seed 42. Release
validation separately checks the stable package. Runtime verification checks
rendering evidence; it does not prove the mathematics.
