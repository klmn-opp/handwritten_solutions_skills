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

After freezing source`884f397383b8af204248d2cd3aa2570f0eab74ad`, fresh regression
[83c8d2ad](../workspace/runs/20261120T162030678252Z-personal-cpu-regression-evidence-83c8d2ad/run.json)
passes24/24 with actual PDF raster pixels, semantic/source identity and English
geometry all exact. It contains2,932 visible mathematical items and0 font fallbacks.
All protected assets, production references and latest remain unchanged. This is
the valid default1.0 compatibility result;28a11fed remains a protected-state failure.

The Agent inspected all four new v4 page PNGs and the same-content line comparisons;
no new clipping, content loss or ink defect was observed. Review receipt:
`workspace/analysis/m22-word-gap-review/FINAL_REVIEW.json`. This review supports
proceeding to author spacing review and exact-package verification, not production.

## Admission

Engineering and author judgments remain distinct. At the v4 checkpoint the prior
output had broad author style approval except spacing; the +6percent candidate was
awaiting author review. The subsequent instruction below resolves that spacing
review requirement. No production/default/latest change occurs.
The frozen4a Mac acceptance baseline remains separate from this later candidate.

## Final author-directed increment: v5

The subsequent exact author response is “再加百分之8即可，后续不用验收这个了”.
Apply8percent to the currentv4 target:1.06×1.08=**1.1448** relative to originalv3.
This is one prescribed amount, not an amount/seed search. The author explicitly
waives another spacing review; do not ask for it again. The waiver covers this
word-gap change and does not authorize publication or production/default activation.

Fresh assets:e7313af3 (`20261120T162900778696Z-m22-word-gap-v5-assets-e7313af3`).
`personal-cpu-candidate@v5` inheritsv4; all192 font/ink/source/recipe asset files stay
byte-identical. Only the outer Profile's render setting and version/lineage change.
ProfileSHA`e7959a8124ba48dc7384c0c1a574bbc6e8da99879c9a8e9809946e69bcc17f81`;
ZIP SHA`06fd786d5df49e2a0c72bdd7e184aa364c8fcd6c55f2a9bc820d786fb02167d2`.
The frozen884 runtime and existing1.0.0rc1 wheel need no code change or rebuild.

The firstv4/v5 comparison79defc9c renders all four pages successfully, with unchanged
formula geometry and passing actual Job checks, but its strict all-glyph-identity
comparison fails. Preserve that failure. There is one real observed secondary effect:
item24, the final `d` in the fourth page title, now selects original recorded d1 instead
of d0. Accumulated word-position changes and pinned Handnote's integer placement give
the preceding `l` 1 px more relative clearance. The unchanged filled-path contact guard
therefore admits the existing lower-bowl d1 proposal that it previously rejected.
There is no new contour, source edit, runtime edit, shape calibration or suppressed
contact gate. It would nevertheless be false to call all output shapes unchanged.

Root independently reviewed the real field comparison and both provenance records,
accepted exactly this existing variant selection, and confirmed clear identity and
the previously preferred lower bowl. The source-bound review is saved under
`workspace/analysis/m22-word-gap-v5-d-decision/root-review.json`. A fresh comparison
requires exactly that item/source change and rejects any extra source or shape
change. This is review of an actual existing-rule consequence, not a new style fit.

Final comparison:
[4f709095](../workspace/runs/20261120T163316655132Z-m22-word-gap-full-four-4f709095/run.json),
[v5 complete PDF](../workspace/runs/20261120T163316655132Z-m22-word-gap-full-four-4f709095/rendered/v5/homework_solutions.pdf),
[v4/v5 local comparisons](../workspace/runs/20261120T163316655132Z-m22-word-gap-full-four-4f709095/comparisons/word-gap-before-after.pdf).
It passes all four pages with0font fallback and unchanged visible semantics/formula
geometry/matrix dimensions. Exactly the independently reviewed d variant changes;
all other sources and contours remain unchanged apart from translation. Page pixels
are exact to the initial fixedv5 render in79defc9c; no candidate tuning occurred.

There are288 shared same-line word pairs:177 widen1px,93 widen2px,1 widens3px and17
remain at their integer position; none shrink. The fourth page's word “radial” moves
to the second prose line as expected from increased spacing; the number of text lines
still remains8/6/10/9. Thus equal line counts do not imply unchanged wrapping. The
new page remains comfortably inside its bounds and retains every word. Source-only
Profile revision uses the identical frozen884 wheel; fresh installed checks must bind
v5 explicitly instead of reusing a v4 Profile receipt.
