# Benchmarks and experimental adapters

The packaged CPU benchmark inventory contains fixed engineering fixtures:

```sh
hwsol benchmark list --suite all
hwsol benchmark run --suite smoke --profile bundled-default@v1
hwsol benchmark --help
```

Keep development, regression and holdout exposure separate. A holdout evaluated
for model selection is consumed and cannot be reused as independent success.
Render-only checks start from supplied answers; end-to-end solving additionally
requires an Agent to read and solve the actual original question. Rendering an
incorrect answer does not verify it. OCR is an imperfect diagnostic, not truth.

`hwsol lab models` and `hwsol lab prepare --help` expose optional experiment
adapters. They are excluded from ordinary rendering and do not make the CPU
runtime depend on Torch. Research workers, private handwriting, GPU environments,
unreleased VAE/model checkpoints, historical experiment results and the developer's
personal math resolver candidate are not supplied in public packages. The audited
`nextgen-v1` inference subset is the only neural model shipped.

Do not expect `lab diffink` or a neural sampling command to work without the
explicit compatible external worker, assets and separate environment it requests.
Some reproduced adapters have vocabulary, prefix or reference-data restrictions;
check the command's validation and upstream documentation. A held result is not
production handwriting. The scoped `personal-neural@1` backend uses the frozen
`nextgen-v1` model for bounded glyph variation only; there is no approved general
text-to-page neural generator in this distribution.

Training and independent generation require separate evidence, fixed data splits,
bounded budgets and complete-page validation. Neither lower loss nor a successful
installation proves personal style, correct continuous writing, or improvement
over a reliable CPU baseline. Preserve failed candidates. Activation, publication
and external sharing of private assets require explicit user approval.
