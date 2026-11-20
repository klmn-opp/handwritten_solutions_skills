# M2.2 author word-gap revision

The author reviewed the final four-page CPU output43bb9000 and the v3 universal-
quantifier comparison, then said: “每个单词的间隔应稍微大一点点，其他整体满意”.
This supports the preceding overall page style, with spacing as the remaining request.
It is not production activation or approval of a spacing revision not yet seen.

## One bounded change

The current candidate uses `semantic_layout=true`. Its actual word-gap control is
`HandwritingLayoutPlanner.text_lines`: it chooses a visible ink-to-ink target in em,
then `space_advance` subtracts left/right glyph side bearings. Changing the font's
space advance would not control this path correctly. Existing Profile assets have
no separate safe spacing override; adding fabricated style observations would mix
layout changes with fake personal measurements.

The sole preselected candidate multiplies this visible target by **1.06**. At the
46px body em and typical0.31–0.34em target this adds roughly0.9px at300DPI, about0.076mm.
It does not multiply letter advances, kerning or MathText token metrics. Existing
punctuation compensation and ordinary line-fit calculation remain in force. Final
visible gaps are quantized by pinned Handnote's integer glyph placement, so the
individual measured increment need not equal exactly6percent of a final outline gap.

`RenderConfig.word_gap_scale` defaults to1.0, with range1.0–1.25. A non-default request
requires semantic layout, and an inactive semantic layout is explicitly rejected.
The only runtime edits are `config.py` and `jobs/handwriting.py`. The public Job JSON
schema and reference document match the new field. Tests use actual public-font
metrics to check the visible target, unchanged nonspace/math widths, default identity
and invalid/ignored-setting rejection. No glyph kernels or source samples change.

## Isolated v4 and evidence

Asset run: `20261120T161608129087Z-m22-word-gap-v4-assets-53c60da6`.
`personal-cpu-candidate@v4` has parentv3. All existing font, ink bank, recipe, nested
Profile and source provenance asset bytes/hashes are exact to v3. Only the outer
Profile render setting, version/creation/description and lineage receipt change.
The archive remains private and inactive. No data collection or neural training.

Input: unchanged original-question intake and corrected full solution Jobab6496e5;
seed1909 is fixed. Both controls and candidates render all four questions through the
normal Job pipeline, with observation-only line and final-vector capture for analysis.
No title, answer, formula, font size or source PDF is rewritten. The control's four
page pixels are required to equal the actual clean-install43bb9000 output.

First full-pair runf62f46ca produces four successful v3/v4 pages, no font fallback,
identical visible semantics and sources/variants. Final per-item path topology is
identical and paths differ only by translation within0.002px numerical tolerance.
English text line counts remain8/6/10/9. Its initial gap statistic also included
separately tokenized punctuation adjacency; retain this diagnostic mistake rather
than treating it as real inter-word spacing. A fresh same-candidate run corrects the
measurement by joining adjacent nonspace punctuation into each word envelope.

Final complete pair:
[0d935793](../workspace/runs/20261120T161842282659Z-m22-word-gap-full-four-0d935793/run.json).
Its v3/v4 page pixels are identical to the first pair; the measurement correction
does not change any rendering input or candidate. All289 same-line inter-word gaps
are retained:206 increase1px,33 increase2px,50 remain at their prior integer placement,
and none shrink. Median increment is1px. No word-pair crosses to another line.
All formula parser-local geometry digests and matrix structures/cell sizes remain
exact; final source/variant changes0 and nontranslation shape changes0. Four pages
pass actual PDF/PNG verification and the unchanged final boundary/contact gates.

Reviewable outputs:
[v4 full PDF](../workspace/runs/20261120T161842282659Z-m22-word-gap-full-four-0d935793/rendered/v4/homework_solutions.pdf),
[same-content local pairs](../workspace/runs/20261120T161842282659Z-m22-word-gap-full-four-0d935793/comparisons/word-gap-before-after.pdf),
and `reports/page-comparison.json` with all measured gaps and checks.
ProfileSHA`4bf72aa8c3ee6c24433f252ac53a89fdc887cd7cea983964cfa4b3624e6ea315`;
archiveSHA`0021af017d39af34733c091d3d68d0286205efcdf9092eb814ea9688624a69c6`.
This is working-source evidence; actual clean RC installation is a separate gate.

The old-default24 regression28a11fed rendered all24 pages but failed its final
protected-asset check because the parent agent changed the root dependency lock's
project version to the local RC during that run. The guard remains strict. This
failed run is retained; repeat only after source freeze. Legacy frozen render configs
omit the new default field, so the harness compares their validated RenderConfig
models; it does not edit old receipts or relax any pixel/source/geometry comparison.

## Admission

Engineering and author judgments remain distinct. The prior output has broad author
style approval except spacing. The +6percent candidate needs review of the new full
PDF and same-content local comparisons. No production/default/latest change occurs.
The frozen4a Mac acceptance baseline remains separate from this later candidate.
