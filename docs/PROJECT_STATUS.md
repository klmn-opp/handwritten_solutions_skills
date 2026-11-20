# Handwritten Solutions — authoritative status

Updated 2026-11-20. Read this page first, then [execution handoff](EXECUTION_HANDOFF.md).
The [user's complete master instruction](MASTER_EXECUTION_PLAN.md) governs future work.
Historical version reports remain historical evidence, not the current status entry.

| Role | Actual implementation | Admission |
| --- | --- | --- |
| Production | main / v0.6.0, Handnote + MathText, CPU; default personal@v1 | Existing stable path |
| Formal latest | v06-five-final, explicitly personal@v3 | Preserved |
| English research backbone | v0.10.21 canonical-conditioned-ink/1, 194,513 parameters; component-balanced checkpoint | Experimental |
| English page calibration | v0.10.22 t/l/r metrics + v0.10.23 measured word gaps | Frozen reference |
| Math research baseline | v0.10.26 D, original personal glyphs + measured structures + spacing/stem correction | M1 fresh-content evidence now available; explicit remaining fallback/refusals |
| Current increment | Third correction delivered: two authentic f sources actually used; isolated pi extraction No-Go | Author review/new natural ink pending; no production promotion |

The first four D pages contain 526 frozen neural English words. The other three
use Handnote English. The seven-page D is not a general neural page service.
The mathematics work has zero new neural updates. Author preference for v24 B
does not constitute author approval of D or an unseen-content evaluation.

Private machine-readable entries are `workspace/registry/production.json`,
`research_best.json` and `candidates.json`. M0 creates them only after verifying
assets and replay; each references its immutable freeze run. Use
`python -m research.project_state verify` to verify existing entries.

History: [2026-11-20 audit](../PROJECT_AUDIT_2026-11-20.md),
[method retrospective](../workspace/analysis/20261120-font-method-retrospective/retrospective.md),
[v26 math report](V01026_MATH_INK_POLISH.md).
Next milestones and decisions: [master roadmap](MASTER_ROADMAP.md),
[decision log](DECISION_LOG.md).

Private scans, Profile snapshots, weight files and active runs remain excluded
from public source and release archives. M2 discovered that broad sdist document
inclusion exposed local operational references; the new explicit public staging
path fixes this without rewriting private working documents or Git history.
An arbitrary raw checkout build is not an approved public distribution.
No production activation is authorized.

M0 verified freeze run: [20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e](../workspace/runs/20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e/run.json).
The C/D replay has 14 identical page images and 58 unchanged protected assets.
[Replayed D PDF](../workspace/runs/20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e/rendered/homework_solutions.pdf).

M0 also re-ran the actual frozen checkpoint: 529 word PNG/float/condition arrays exactly match.
See [M0 evidence and limitations](M0_FREEZE_REPORT.md). No new neural training occurred.

## Actual new-content and data results

[M1 homework report](M1_HOMEWORK_BENCHMARK.md) is the current page/coverage entry:
24 newly Agent-authored problems, 13 successful development pages and three expected
refusals, seven successful one-use holdout pages and one unexpected explicit refusal.
The original holdout remains **7/8**. Its matrix-spanning brace expression has a
separate equivalent-source recovery; it is not a newly supported renderer structure.
Four different original-text problems were actually solved, checked and rendered
through the public CPU path. No new image/PDF intake evidence is claimed.

The 20 original successful A/B pages contain 2,269 visible math items. Font fallback
occurrences decrease from 201 to 38, with zero changed English pixels. Root Agent
review found no confirmed content/boundary error in accepted pages. The author then
rejected visual acceptance: long/tall root clearance, slash fractions, heavy title
zeros, r spacing, high d/k bodies and small/faint f require correction. The active
[author revision](M1_AUTHOR_REVISION.md) preserves the original evidence.
Both fresh-content arms use CPU English. These results do not
establish fresh neural English generalization or arbitrary math coverage.
The first complete author revision has 2,265 visible math items and 30 font
fallbacks; the author confirms clear improvement but identifies heavy 2/pi,
printed-looking fraction/root bars and short/faint f. The completed
[second correction](M1_SECOND_AUTHOR_REVISION.md), run `956bbad6`, changes92 twos,
129 fraction bars,14 radical roofs and115 f. All20 pages received Agent visual
review; non-target pixels/geometry/provenance stay unchanged.20 pi,30 f,7 short
roofs and30 font fallbacks remain explicit. Author feedback now likes the f shape
but identifies collapsed f variety and retained heavy pi. The completed
[third correction](M1_THIRD_AUTHOR_REVISION.md), run `4af6334e`, uses actual sheet-f1
in 53 places and f0 in 92, preserving the accepted proportions, all 14 ff groups,
the original 30 held f and all non-f output. All final PNGs equal the reviewed
parent. A lighter historical pi is genuinely connected to its preceding 2, so
independent extraction remains No-Go. The source search also exposed a legacy
nested-holdout label-access flaw; the actual shared reader is fixed and the
15-label access is disclosed, without C access or original-image use. No new
neural training occurred. The
[comprehensive 19-page collection booklet](COMPREHENSIVE_MATH_COLLECTION.md)
now supersedes the minimum-collection preference and is ready to fill.
See the reports for PDFs, PNGs, all 173 formula pairs and fallback overlays.

[M3 corpus report](M3_CORPUS_READINESS.md): immutable federation and incremental
replay are implemented, retaining 11 documents, 297 overlapping original region
observations, 609 source/derivative views and 679 dataset memberships. The 174 original
words, 39 lines, 275 current local targets and 84 math regions are not additive
independent training samples. Same-source replay adds zero observations. Existing
exposure and review status are preserved; no new author holdout has been created.

[M4 mechanism preflight](M4_MECHANISM_PREFLIGHT.md): no new training. On the old
18 true-condition word reconstructions, exact zero-support constraints explain only
0.000834% of L1 error; local contour/ink and missing morphology conditions dominate.
This rejects the proposed justification for unconstrained ink addition or simply
repeating training on the same 13 complete words. The subsequent
[context source review](M4_CONTEXT_DATA_AUDIT.md) found authentic continuous ink
beyond those words. The [16-span alignment pilot](M4_ALIGNMENT_PILOT.md) replayed
all original crops exactly and ran 32 real frozen A/B0 forwards, with unchanged
weights. The proposed local residual still conflates morphology, geometry and tone;
its A/B/C training remains No-Go. No author engineering work is required by this
negative finding, and no new neural improvement is claimed. RL remains No-Go.
