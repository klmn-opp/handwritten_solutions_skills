# M4 mechanism preflight — zero updates

2026-11-20. **No-Go for another context training run on the old 13-word set.**
The exact-zero limitation of `WordContextInk` is mathematically real, but it does
not explain the visible errors in these 18 words. Their main measured residuals
are local contour/ink differences and, in five words, missing reliable morphology
conditions. The evidence also does not support treating these residuals as a
whole-word placement problem or as missing connecting strokes.

This is an analysis of previously exposed **true-condition reconstruction**.
No model was trained or run for new inference. No independent word generation,
author similarity improvement, new genuine handwriting or production admission
is claimed. The source-conditioned boxes and morphology already contain much
more information than a new-text generator receives.

Evidence:
[20261120T105039208020Z-context-mechanism-preflight-c50fd8fd](../workspace/runs/20261120T105039208020Z-context-mechanism-preflight-c50fd8fd/run.json).
[All 18 comparisons, PDF](../workspace/runs/20261120T105039208020Z-context-mechanism-preflight-c50fd8fd/comparisons/mechanism-diagnosis.pdf)
· [overview PNG](../workspace/runs/20261120T105039208020Z-context-mechanism-preflight-c50fd8fd/comparisons/all-18-overview.png)
· [per-word measurements](../workspace/runs/20261120T105039208020Z-context-mechanism-preflight-c50fd8fd/reports/words.json)
· [source records](../workspace/runs/20261120T105039208020Z-context-mechanism-preflight-c50fd8fd/reports/original-sources.json).

## Frozen inputs and method

- Canonical audit: `20261120T023639068683Z-canonical-intact-word-audit-627f2eae`;
  18 intact real words, existing 128×512 targets, boxes, true local conditions and
  frozen reconstruction arrays. Thirteen have all positions matched; five do not.
- Frozen v21 backbone: `20261120T022439649806Z-canonical-ink-component-refinement-fbea19c9`,
  checkpoint SHA `779a2ab51902a17a7068df2c32bf7f5ceddef938ee6982f116f39e0dfb3ad15d`.
  Its saved 96×96 local reconstruction arrays are reused for geometry diagnostics.
- Historical v22 context output:
  `20261120T030504700173Z-bounded-real-word-context-training-9bae57b9`.
  Reading that existing output does not perform another 1,000 updates.
- Original RGB word crops are copied byte-for-byte and SHA-checked. Original
  labels, document/page coordinates, transforms and prior reviews are retained.
  The 18 words come from four original documents; they are all exposed Train
  material. Their 13 complete-condition subset also trained the old adapter.

The executable is
[`research/context_mechanism_preflight.py`](../research/context_mechanism_preflight.py),
outside the frozen renderer source tree. It uses NumPy/Pillow and the existing
immutable `Run` contract, requires no Torch, and saves its own source plus the
inspected odds/composer source. Inputs, original crop hashes, checkpoint and
formal latest are checked before/after. Session C and protected siblings are not
read. Runtime and hardware are in `run.json`.

Actual execution and reproducibility command:

```bash
.venv/bin/ruff check research/context_mechanism_preflight.py
.venv/bin/python -m research.context_mechanism_preflight
```

Every invocation allocates a new private run. Two intermediate diagnostic runs
remain preserved: the first exposed why intersecting overlapping character boxes
is an invalid ligature detector; the final run requires independent character
anchor regions and records the earlier overlapping-box warnings explicitly.

## What the endpoint restriction actually prevents

The existing adapter computes

`p' = p + p(1-p)(exp(delta)-1)/(1+p(exp(delta)-1))`, with `delta` in `[-3,3]`.

Thus exact `p=0` and `p=1` cannot change. For other pixels the attainable interval
is `[p/(p+(1-p)exp(3)), p/(p+(1-p)exp(-3))]`. Clipping the true target into this
interval gives a pointwise best-case error floor. This **oracle uses the answer
image at every pixel**; it is an expressivity bound, not a model, achievable
learning result or usable generator.

| Measurement | 13 complete-condition words | 5 incomplete-condition words |
| --- | ---: | ---: |
| Original L1 sum | 11,345.91 | 8,885.16 |
| Error locked at exact zero/one | 0.16078 | 0.00784 |
| Locked fraction of original L1 | 0.00142% | 0.000088% |
| Foreground pixels locked at exact zero | 0 | 0 |
| Any exact-one input pixels | 0 | 0 |
| Best possible ±3 odds L1 floor | 356.27 (3.14%) | 2,493.51 (28.06%) |
| Foreground unreachable above 0.5 even at +3 | 2 | 1,253 |
| Extra ink unremovable below 0.5 even at −3 | 16 | 1,036 |

Across all 18 words, exact zero locks only 21 faint target pixels with total ink
0.16863 and maximum intensity 0.01569. It accounts for **0.000834% of L1**, and
locks no target foreground under the declared 0.5 threshold. The historical
adapter preserves the exact zeros as expected.

This corrects the tempting causal inference from the M3 architectural observation:
“cannot create ink at zero” does not imply that zero support caused these errors.
The five words with missing morphology conditions account for 87.5% of the bounded
odds floor. Raising the odds range or adding a free residual solely because the
formula has fixed endpoints would not address the demonstrated data/condition
problem and would increase opportunities for incorrect ink.

## Where the errors occur

All 5,230 false-negative foreground pixels lie inside the registered character
boxes. The positive-width gaps between character boxes contain **zero** target
foreground and zero missing foreground. In the complete subset, 2,144/2,215
missing pixels (96.8%) lie within two pixels of an existing predicted stroke;
only 71 are farther than two pixels and 37 farther than four. There are also
5,558 extra foreground pixels, versus 2,215 missing ones. This is primarily an
edge/ink-shape diagnostic, not evidence that the author expects blank spaces to
be filled with new connecting strokes.

The five incomplete words have explicit zero morphology vectors at eight
positions, because their original cuts were not admitted:

| Word | Missing reliable condition positions (zero-based) | Missing-box L1 / whole-word L1 | Missing-box false negatives / word total |
| --- | --- | ---: | ---: |
| maximum | 2:x, 3:i | 1,910.31 / 3,284.89 | 890 / 1,126 |
| boundary | 7:y | 935.18 / 1,548.43 | 257 / 466 |
| can | 0:c, 1:a | 1,489.98 / 1,804.93 | 438 / 515 |
| large | 3:g | 811.75 / 1,092.05 | 399 / 499 |
| Total | 2:t, 3:a | 808.76 / 1,154.85 | 347 / 409 |

These unions of suspect boxes contain 67.0% of the five words' L1 and 77.3% of
their false negatives. This localizes an association; it is not an isolated causal
experiment that supplies the missing true conditions. Complete whole-word labels
remain valid. Their unadmitted single-letter cuts must not be restored simply to
make an experiment convenient.

Visual inspection of the original RGB and all-word overview supports this split.
The incomplete `maximum` has a conspicuous x/i morphology mismatch;
[`large`](../workspace/runs/20261120T105039208020Z-context-mechanism-preflight-c50fd8fd/comparisons/word-15.png)
and [`Total`](../workspace/runs/20261120T105039208020Z-context-mechanism-preflight-c50fd8fd/comparisons/word-17.png)
show disagreements around their missing-condition positions. The complete
[`lies`](../workspace/runs/20261120T105039208020Z-context-mechanism-preflight-c50fd8fd/comparisons/word-14.png)
still has local fine-stroke/contour differences, including its i accent and s.
These are Agent diagnostic observations, not a new author style rating or new
training annotations.

## Alignment and actual continuous-ink support

A target-informed integer translation search in x/y `[-2,2]` improves the complete
13-word L1 by **0%**. A one-pass per-character search over the same shifts improves
its own recomposed baseline by **0.0717%**; only four one-pixel vertical changes
are selected. Thus simple placement error is not the dominant residual in this
source-aligned reconstruction test. More general affine or nonlinear alignment
has not been ruled out, and new-text page layout is a different test.

Per-character geometry uses saved local outputs and a NumPy implementation of
the recorded bilinear composer. It is not bit-identical to the GPU audit: mean
absolute discrepancy is at most 0.0000286 per word, maximum pixel discrepancy
0.002091, and there are five threshold disagreements across the 13 words. Geometry
improvement is therefore measured against its own unshifted recomposition; the
endpoint/error tables use the original saved audit floats. Both geometry outputs
and odds outputs in the PDF are clearly labeled target-informed **not generation**.

A connected component touching two overlapping boxes is insufficient to establish
a ligature. It can be the same single-character stroke counted twice. That naive
rule flags `xi`, `ry`, `ca`, `rg`, and `ta` in the five incomplete words. After
requiring at least five connected foreground pixels in **each character's region
outside the other character's box**, none of the 18 words has a qualifying
cross-character component. This result is unchanged at target thresholds 0.25,
0.5 and 0.75. The RGB previews do not reveal a clear missing cross-gap join that
would reverse this finding.

This is a bounded negative finding on these samples, not a statement that the
author never joins letters, nor a recovery of actual pen trajectories. Whole-word
supervision remains useful for rhythm, shape and composition even when letters
are disconnected. However, these 18 words do not substantiate the proposed claim
that a new additive stroke module is needed to learn actual joined handwriting.
Forcing joins on predominantly disconnected personal writing would be a style
regression.

The old adapter reduced complete-word L1 by 20.9%, but the prior frozen neighbor
ablation found mean neighbor effect 0.000060 versus total correction 0.007566.
That separate, already exposed word/line experiment supports a mainly local
sharpness interpretation. Neither its reconstruction gain nor the target-informed
oracles establish independent continuous generation.

## At most one next hypothesis, conditional on data

**H-context:** an adapter trained on reviewed original adjacent-ink relationships
that are demonstrably unavailable to the independent-character representation
can improve those relationships on new text, beyond a word-shared-state control,
without introducing any confirmed character/content error.

The changed ingredient must be the demonstrated contextual supervision and its
matched representational support. Another 1,000 updates on the same 13 clean
reconstructions does not test this hypothesis. The present preflight does **not**
meet its data premise and therefore does not authorize starting that experiment.

Minimal future A/B/C implementation, if the data premise is met:

| Arm | Implementation boundary | Necessary control |
| --- | --- | --- |
| A | Frozen v21 local generator and existing composer/layout. | Fixed text, per-letter content, seed, Profile and math layout; preserve all outputs, including failures. |
| B | A plus one shared word state with class-aware mappings restricted to supported morphology axes. | Never copy the same raw PCA vector across incompatible character bases. No cross-character ink residual. A zero-state pass must exactly recover A. |
| C | The same B state plus a small semantic/neighbor-conditioned residual, supervised by intact original words or reviewed fragments. Its support is restricted to regions for which real contextual evidence exists. | Include a neighbor-removal ablation with identical parameters. Any ability to add ink into zero support must be explicit and content-gated; no unrestricted page/word image generator or invented connecting target. |

Before freezing such a protocol, search the already registered 174 original words
and the 39 accepted lines for the `th/st/ll/ing/ff/fi/ti` opportunities listed in M3.
Keep complete source regions. Confirm actual adjacent-ink relationships and labels;
do not treat a substring, touching boxes, or unsafe single-letter crop as proof.
A small feasibility gate can require at least 20 reviewed complete words/fragments,
at least three source groups, and at least two target context families represented
in more than one group. If those cannot be found, report the actual gap before
requesting minimal new personal writing. The exact supervised task should follow
what the author really writes; the audit must not demand joining where none exists.

A bounded proposal after that gate is one seed and at most 500 updates each for
B and C, frozen v21 weights, fixed source-group assignment, fixed final checkpoint,
and no opportunistic seed/step selection. The finite budget is a stop condition,
not a quality target. Freeze the detailed data/model/loss protocol before training.
Use whole-word/fragment pixels for C's supervision and preserve unmatched character
positions without pretending their local codes are known.

Stop and retain No-Go if C has any confirmed new content defect, if apparent
improvement is only training reconstruction or local sharpening, if neighbor
removal leaves essentially the same result, or if it does not beat B on the
predeclared adapter-development comparison. A provisional diagnostic threshold
can require at least 10% contextual-error reduction and neighbor contribution
above 5% of total correction, but neither proxy can independently accept the
model. Validate fixed new-text words/lines and complete homework pages; author or
appropriately calibrated independent assessment is still required for M4 success.
Thresholds must be frozen before that future experiment, not chosen from outcomes.

All proposed new-adapter splits remain exposed to the upstream backbone. Session
B's prior 10/3 complete-word adapter partition is not a whole-system holdout, and
this analysis has further consumed all 18 words as development evidence. New text
prompts are not new authentic author references. Sealed C stays sealed; no new
holdout protocol was activated or altered.

## Decision and next work

**Go:** keep the frozen v21/Math Resolver combination; continue M1/product gates;
perform source review of historical contextual examples and quantify their actual
coverage. This work does not presently require author intervention.

**No-Go:** repeat old 13-word context training, launch an additive-ink model based
on the exact-zero argument, claim these oracles as generated handwriting, or
promote a neural model. This stage performed zero new neural updates and zero new
network inference. It resolves the mechanism question with inspected original
ink and reproducible negative evidence rather than manufacturing a new training
milestone.
