# M4 intact-span alignment feasibility pilot

2026-11-20. **The bounded pilot is complete; the proposed local A/B/C training
protocol remains No-Go.** It establishes a usable source-coordinate chain and a
real zero-state inference contract. It does not establish a new handwriting model
or a contextual style improvement.

The reason is identifiability, not an IoU/L1 threshold: independently sampled
local shapes are not paired with the original word's morphology. Word geometry,
pen weight, source background and natural within-author variation still contribute
to the same apparent span residual. The current experiment cannot reliably assign
that residual to context. The real continuous words remain useful supervision;
this result neither rejects whole-word/context learning in general nor requires
the author to do engineering alignment work.

## Immutable evidence and scope

Initial source annotations and **32 actual new GPU forwards**:
[`20261120T111838309635Z-context-alignment-pilot-88af1c54`](../workspace/runs/20261120T111838309635Z-context-alignment-pilot-88af1c54/run.json).

Final reviewed annotations and sensitivity analysis:
[`20261120T112242160369Z-context-alignment-review-ba28f733`](../workspace/runs/20261120T112242160369Z-context-alignment-review-ba28f733/run.json).

The final child explicitly reuses the immediate parent's newly computed float
arrays; it performs zero additional network calls. These are not cached historical
word images presented as fresh inference. Its `lineage.json` retains parent and
inference-manifest hashes. The first annotation sheets remain in the parent run.

Review artifacts:

- [Four-page original RGB span review](../workspace/runs/20261120T112242160369Z-context-alignment-review-ba28f733/comparisons/roi-review.pdf)
  and `roi-01.png` through `roi-04.png` in the same directory.
- [Four-page original/A/B0/geometry comparison](../workspace/runs/20261120T112242160369Z-context-alignment-review-ba28f733/comparisons/inference-review.pdf)
  and `inference-01.png` through `inference-04.png`.
- [Full source records and semantic ROI annotations](../workspace/runs/20261120T112242160369Z-context-alignment-review-ba28f733/reports/rows.json),
  [PDF coordinate replay](../workspace/runs/20261120T112242160369Z-context-alignment-review-ba28f733/reports/pdf-transforms.json),
  [actual inference calls](../workspace/runs/20261120T112242160369Z-context-alignment-review-ba28f733/reports/inference.json),
  [per-word measurements](../workspace/runs/20261120T112242160369Z-context-alignment-review-ba28f733/reports/measurements.json).
- All 16 untouched RGB word files, 16 intact RGB span slices, fresh layout tokens
  and boxes, 32 A/B0 PNGs, float outputs, sampled conditions and diagnostics are
  retained in `solutions/`, `rendered/` and `comparisons/`.

The fixed sample contains 16 word observations, 12 distinct exact texts and five
source groups: A, B, Physics, Calculus 1 and ICS. These are previously reviewed
development materials; all source documents were exposed upstream to v21. No
independent author-data holdout or unseen-text success is claimed. No sealed C,
additional document search, new transcription or neural training was used.

## Source and ROI audit

The pilot uses exactly the 16 spans listed in
[M4_CONTEXT_DATA_AUDIT.md](M4_CONTEXT_DATA_AUDIT.md). Original word labels,
whole-word crops, review history, session/document groups and old splits remain
unchanged. The new Agent annotations identify complete `ff`, `fo` and `th`
regions, with no forced single-letter segmentation and no reconstructed strokes.

Thirteen semantic rectangles can serve as locally bounded diagnostic regions.
`w017 coefficient`, `w006 the` and `w101 they` include neighboring e ink or a
connection across the region boundary; these remain **whole-word-only** for
supervision. Blue rectangles are semantic areas with background retained, not
pixel-perfect foreground labels. Orange rectangles deliberately expose ambiguity.

The root Agent inspected all four first-round RGB sheets and accepted these
conservative classifications for feasibility, without promoting them to training
truth. Native-pixel reinspection then extended the right edges of `w070` from
117 to 120, `w093` from 88 to 90 and `w118` from 68 to 70 pixels, retaining cap/tail
antialiasing without cutting those complete spans. The child keeps the old and new
rectangles. The final sheets add page and cell margins; the first rendering remains
available rather than being overwritten.

For every word, the audit restored `word → parent crop → 300 DPI page raster`.
It independently rendered only the corresponding approved PDF pages with PDFium
and recovered `page raster → PDF` using the renderer's actual coordinate converter,
including page rotation and crop geometry. **All 16 rendered word crops match the
retained original RGB pixels exactly.** Both affine transforms and native PDF/page
metadata are saved. No nominal 72/300 conversion substitutes for this verification.

## Actual A/B0 inference contract

A uses the frozen `canonical-conditioned-ink/1` decoder, 194,513 parameters,
checkpoint:

`workspace/runs/20261120T022439649806Z-canonical-ink-component-refinement-fbea19c9/logs/checkpoint-01000.pt`

SHA-256: `779a2ab51902a17a7068df2c32bf7f5ceddef938ee6982f116f39e0dfb3ad15d`.

Layout uses unchanged `personal@v1`, 300 DPI, 11 pt, line spacing 1.6, seed 1909,
raw glyph metrics and semantic layout. Class aspect calibration and t/l/r spacing
thresholds come from the frozen v0.10.22–23 mixed-homework request and its metric
parent, copied into the new request. The new twenty-page mathematical feedback
was not incorporated into these conditions.

The sampler resets seed 2012 + 200000 for each word, selects a same-character
training PCA center and adds 0.15 times that class's standard deviation in Gaussian
noise. Thus repeated identical texts receive the same sampled conditions, while
each occurrence still causes an independent actual forward. All characters in
this pilot have real-sample class centers; none used the font-only zero-code path.
Text identity and fresh layout geometry are inputs. Target RGB pixels and exact
target morphology codes are never passed to the network. This is independent
class-conditioned sampling, **not true-condition reconstruction**.

B0 calls the decoder separately with the same conditions after an explicit shared
physical bounding-box state: log scale x/y, shear and baseline displacement, all
zero. It owns no learned parameters. It is only the zero-state contract of a
possible future word-state interface; it is **not a trained shared-context B**.

The real GPU run recorded 32 model forward-hook events. All 16 A/B0 float arrays
are bit-identical (`max_abs_difference = 0`). Torch gradients were disabled;
no optimizer was created; in-memory parameters and the checkpoint hash were
unchanged. GPU: NVIDIA GeForce RTX 5060 Laptop GPU; the 16-word inference loop
took approximately 0.60 seconds. This timing excludes preparation/import/model
loading and is not a CPU or end-to-end performance claim. Exact Torch/CUDA
versions and memory are in `inference.json`.

## What the diagnostics show

Original whole-word layout is retained. Target and generated images are each
uniformly fitted into 512×128 using the existing whole-word normalization recipe;
integer resize rounding and invertible mappings are recorded. No per-character
target warp is introduced. The original RGB remains separate and unmodified.

A fixed 27-case sensitivity grid applies only whole-word scale 0.97/1/1.03 and
translations −2/0/2 pixels. Its chosen transform minimizes whole-word L1 and is
explicitly an **oracle diagnostic using the target**, not a generation capability.
It is applied equally to A and B0, which are equal. No seed selection occurs.

| Diagnostic | Observed value | Interpretation limit |
| --- | ---: | --- |
| Semantic span-box IoU, median | 0.582 | Box correspondence only, not handwriting quality |
| Whole-word L1 decrease under global grid, median | 6.10% | Modest geometry sensitivity, not a learned gain |
| Span L1 decrease with that same transform, median | 3.32% | Four spans become worse despite whole-word improvement |
| Red `for` target foreground 90th-percentile ink | 0.573 / 0.569 | Grayscale extraction changes apparent contrast |

The ±3%/±2-pixel grid does not exhaust possible word-level geometry differences.
Its residual cannot be called context-only merely because a small transform fails
to remove it. The generated/source rectangles are semantic bounds, not corresponding
stroke landmarks; a box IoU threshold does not admit or reject a model.

Visual comparisons show that source f can have a deeper descender, loops and
crossbars differ, and word widths/x-height ratios differ before any hypothetical
context module. Some generated th strokes already touch through composition while
some authentic controls do not. Connection itself is not the desired universal
target. Original Physics words are much heavier than the fine Calculus sources;
red ICS ink is attenuated by the existing grayscale formula. A separate target
P90 gain diagnostic changes the red span L1 from 0.176 to 0.202 and 0.167 to 0.189
without changing target geometry. This is diagnostic tone scaling only, not a
replacement training image or repaired original.

The particularly useful within-document control is ICS page 7: joined `w170 for`
and separate `w113 for`. They share source group, exact text and fixed generated A,
yet exhibit different original behavior. Their normalized source L1 is 0.00963.
Three `the` observations likewise have identical fixed generated A but substantial
original geometry/weight differences. These controls establish natural variation;
they do not identify which unobserved latent state produced each real sample.

## Go/No-Go and next boundary

**Go:** source-coordinate registration, intact semantic span review, genuine
frozen-model inference, and the zero-state contract are demonstrated. The originals
can support whole-word proportion, rhythm, start/end and natural-variation work.

**No-Go:** training the currently proposed local A/B/C residual hypothesis. C has
not been implemented or given a defensible matched residual target; there are no
placeholder C outputs. The pair-region mask alone does not remove morphology,
layout, pen and background confounding. This is the conclusion from the actual
input contract and images, not a proxy metric passing or failing a numerical gate.

Any later controlled protocol must first specify what B/C learn (e.g. shared
geometry versus local ink), how those effects are separated using whole-word and
natural-separation controls, and how independently generated outputs are evaluated
without target-specific fitting. It must preserve grouped development exposure and
report reconstruction separately if any matched morphology is later used. Whole-word
supervision remains an option and does not inherently require cutting joined letters
or supplying target shape at inference. No further experiment or collection is
started by this report; the root Agent requested this pilot be sealed here while
the independent twenty-page visual repair proceeds.

No author intervention is needed for this engineering result. Production defaults,
latest, original sources, old Profiles/data and model weights are unchanged. No
new neural training, generalization improvement or personal-style victory is claimed.

## Execution and reproducibility

Implementation: [research/context_alignment_pilot.py](../research/context_alignment_pilot.py).
The CPU entry points do not import Torch; only `worker` and its shared-state
helper do. Commands and environment attempts are retained in the final run's
`logs/commands.txt`, with source snapshots for preparation, actual inference,
review and analysis.

```bash
.venv/bin/python -m research.context_alignment_pilot prepare
.venv/bin/python -m research.context_alignment_pilot pdf-geometry <initial-run>
MPLCONFIGDIR=/tmp/hwsol-mpl-alignment \
PYTHONPATH=src:.:workspace/environments/english-htr-v08/lib/python3.12/site-packages:.venv/lib/python3.12/site-packages \
workspace/environments/english-htr-v08/bin/python \
  -m research.context_alignment_pilot worker <initial-run>
.venv/bin/python -m research.context_alignment_pilot revise <initial-run>
.venv/bin/python -m research.context_alignment_pilot analyze <review-run>
.venv/bin/python -m research.context_alignment_pilot seal <review-run>
```

The isolated GPU environment lacked the CPU project's Pydantic installation, so
read-only CPU dependency paths were added after the GPU environment's packages;
no dependency was installed or upgraded. The sandbox GPU attempt could not access
CUDA; the already authorized isolated inference then ran with reviewed GPU access.
The failed import/probe attempts performed no network forward or parameter update.
Both Python environments are 3.12.16.

Validation includes exact RGB/PDF replay, model-call and no-update checks, source
and parent-array hashes, affine round trips, explicit visual inspection of every
source span and output, and Ruff. These checks establish provenance and experiment
integrity, not style quality or training progress.
