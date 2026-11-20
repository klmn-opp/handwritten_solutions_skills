# M3 Corpus 2.0 readiness and M4 data admission

2026-11-20. This is an audit and source registration result, not completion of the full
Corpus 2.0 product or a new training result. It follows the
[master instruction](MASTER_EXECUTION_PLAN.md) and preserves all existing data splits,
labels, Profiles, runs and checkpoints. No Session C images or protected sibling
crops were opened. No new human labels, original handwriting or neural updates
were created.

Evidence run:
[20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8).
The run stores the executable audit in `logs/corpus-readiness-audit.py`, its source
inputs, protected file hashes, grouped registries and decisions. It is sealed `held`
because the inventories are engineering evidence, not acceptance of model quality.

## What is actually available

| Asset | Verified scope | Use and limitations |
| --- | --- | --- |
| Reviewed English lines | 39 lines, 8 documents, 17 pages; 200 whitespace-separated transcription tokens | All have accepted Agent visual review. They are existing Train material, not human-gold transcription or a fresh holdout. |
| Original word links | 158 distinct RGB word crops linked to those lines | Every crop was recomputed from its saved original parent region and matched exactly. The 42 remaining line tokens are not automatically additional isolated training words. |
| Adjacent word geometry | 110 pairs across 38 lines | Suitable for measured rhythm supervision; pairs are observations within the same lines, not independent new samples. |
| Current local glyph targets | 275 views, 28 actual-scan character classes, 68 original parent words, 65 distinct parent transcriptions, 8 documents, 14 pages | Same sources used by the frozen v0.10.21 backbone. Derived views and density-adjusted views do not count as new original samples. |
| Font-only targets in current generator | 44 character classes | Explicit auxiliary targets, including all ten digits and some letters/punctuation. Personal math originals do not automatically become neural English training. |
| Strong intact word targets | 18 whole words from 4 documents, with verified character positions | All 18 overlap the glyph-parent set; 15 also belong to the 158 registered words. Only 13 have a reliable local morphology condition for every character. |
| Larger historic word dataset | 613 views: 553 Train and 60 old Validation | Train contains 535 whole words, 7 fragments, 11 auxiliary glyph views. The 60 Validation words come from one historically exposed source document; they are not a new final test. This is a mother dataset, not 613 new additions to the smaller collections. |
| Historical math extraction | 19 regions, 18 accepted, 18 labels | Original pixels, context, source PDF and transforms retained. One historical candidate remains held. |
| Sessions M/N | 65 regions, 58 accepted, 34 labels, 7 held | Two source pages with original RGB crops and transforms. These were geometry/contour calibration, not gradient training. |

The accepted historical and M/N math label sets have 4 labels in common and
48 distinct labels in their union. That union excludes the older Profile's other
characters and does not assert arbitrary expression coverage or independent
styles for each size.

The retained original assets were checked against 667 unique SHA-256 entries.
The 275 local training images contain no exact duplicate files or identical decoded
gray images. This is not an exhaustive near-duplicate analysis. Known shared source
and derivation links remain the required grouping key regardless of pixel difference.

[Machine summary](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/result.json)
· [asset checks](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/asset-checks.json)
· [original document records](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/original-documents.json)

## Source grouping and exposure

| Original document/session | Reviewed lines | Registered words | Adjacent pairs | Local glyph views | Glyph parent words | Strong words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Session A | 9 | 45 | 36 | 144 | 35 | 12 |
| Session B | 5 | 26 | 21 | 91 | 19 | 4 |
| ICS | 1 | 3 | 1 | 19 | 5 | 0 |
| Calculus 1 | 8 | 27 | 17 | 1 | 1 | 0 |
| Calculus 2 | 3 | 8 | 5 | 7 | 2 | 0 |
| Calculus 3 | 1 | 2 | 1 | 7 | 2 | 1 |
| Physics 2 | 11 | 44 | 28 | 3 | 3 | 0 |
| Formula Sheet I | 1 | 3 | 1 | 3 | 1 | 1 |

All eight documents are represented in the frozen local backbone's training.
Sessions A/B account for 235/275, or 85.45%, of the local views. Document diversity
therefore does not imply balanced style supervision. Fifty-two of the 68 local
glyph parent words also occur among the 158 linked words.

The base SQLite corpus and the later imported-session SQLite snapshot are separate
catalogs. Newer line, word and glyph reviews live in immutable research run
manifests. M/N source entries have their own PDF hashes and coordinate lineage,
but no legacy SQLite document/page IDs. The current hierarchy is a traceable
federation of registries; a unified incremental Corpus 2.0 interface is still work
to do. The new audit registers common source groups without rewriting any catalog.

The current `session:<document>` keys preserve capture/document grouping. For old
course PDFs they do not prove the actual handwriting date or that two PDFs came
from independent capture sessions. Where real session identity is unavailable,
retain that uncertainty instead of asserting stronger independence.

[Grouped sources](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/documents.json)
· [line registry](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/line-registry.json)
· [word registry](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/word-registry.json)

## Character and continuous-ink coverage

Current actual-scan local counts are:

```text
A1 F1 H1 S1 T3
 a20 b2 c4 d6 e17 f3 g3 h7 i32 l19 m16 n34 o13 p12 q1 r16 s22 t15 u17 v1 w6 y1 z1
```

Lowercase `j`, `k`, `x`, most capitals, all digits and some punctuation remain
font-only in this specific local generator. `x` appears in whole-word supervision
but has no safe isolated crop admitted to the current local targets.

| Character | Initial | Medial | Final | Actual reliable isolated total |
| --- | ---: | ---: | ---: | ---: |
| f | 2 | 1 | 0 | 3 |
| i | 4 | 28 | 0 | 32 |
| t | 2 | 6 | 7 | 15 |

The three safe `f` crops come from `final`, `for`, and `unfiltered`. The 158 linked
whole words contain textual opportunities `th` 31, `st` 5, `ll` 3, `ing` 2, `ff` 2,
`fi` 3, `ft` 0 and `ti` 6. These are substring occurrences, not confirmed connected
ligature counts. Their full original words can be reviewed without forcing cuts.
The 18 strongly positioned words contain only four `th` opportunities and one
`fi`, with none of the listed `st/ll/ing/ff/ft/ti` sequences. They alone cannot
validate broad natural joining.

[Sequence opportunities and limits](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/text-sequence-opportunities.json)

## M4 admission and the smallest meaningful comparison

A bounded development experiment is possible with current material once the
parent M0/M1 engineering prerequisites and a distinct hypothesis are frozen.
Independent final personal-style acceptance is not yet ready.

| Arm | Usable starting point | Required distinction |
| --- | --- | --- |
| A, local baseline | Frozen v0.10.21 component-balanced checkpoint and the registered composition/layout | Reproduce independent style sampling, not only true-condition reconstruction. Keep data/condition identities for every generated character. |
| B, word-shared state | Existing real local codes grouped by their original word and position | Different characters have different PCA bases. Sharing the same raw 8-vector across classes does not establish shared physical style. Use class-aware mappings, supported axes and a zero-update control. |
| C, real continuous supervision | 18 original intact positioned word targets; 13 are fully condition-matched. Broader 158 word links and 39 lines remain genuine original continuous targets | C must use whole-word image loss with reviewed labels and explicit geometry confidence. Do not invent missing cuts or treat estimated character positions as ground truth. Repeating the old 13-word v22 experiment without a new mechanism is not a new hypothesis. |

A concrete source-separated **new-adapter development** partition is saved as a
proposal: keep all Session B derivatives out of a new adapter's training.
This provides 14 strong words for training and 4 for optimization validation,
or 10 and 3 respectively if every character must have a known local condition.
The line partition is 34 training and 5 optimization-validation lines; associated
words, fragments and glyph derivatives must follow the same whole-document split.
This does not alter the existing dataset's Train labels.

The frozen backbone and its PCA already saw Session B. This partition is therefore
not a whole-system unseen-document holdout. No evaluated development prompt may
be rebranded as a new independent test. New text prompts can test unseen text
combinations after overlap checks, but they do not supply an unseen authentic
writer reference. A final style test needs a frozen candidate/protocol and a
separate appropriate author comparison or calibrated independent evaluation.
The sealed Session C remains unopened.

[Partition proposal](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/adapter-partition-proposal.json)
· [strong word targets and original transforms](../workspace/runs/20261120T102500322884Z-m3-corpus-readiness-audit-bfc657e8/reports/strong-word-registry.json)

One architectural issue is provable directly from the existing code.
`WordContextInk.forward` computes
`base + base * (1 - base) * e / (1 + base * e)`.
At exact zero or one input its output is unchanged. This odds-only correction
cannot create a connecting stroke in an exactly blank inter-character gap.
Consequently it cannot substantiate a hypothesis of new joining support without
an explicitly different, bounded and content-checked mechanism. Existing
v22 context code is still useful as a sharpness control.
[Implementation](../research/corpus/word_context_model.py)

## Concrete next data tasks

1. Build an incremental registry adapter that reads the existing catalogs and
   accepted run manifests, emits source-group and derivation edges, and preserves
   each review's author/model provenance. Use source hash plus source region and
   transform, not a newly assigned image filename, as the identity anchor.
2. Review existing complete words containing `st/ll/ing/ff/fi/ti` before making
   any collection request. Retain complete ink and mark join existence and local
   alignment confidence separately. Textual opportunities in this report are
   candidates, not a new approved ligature dataset.
3. Search already authorized Train documents for additional medial and final
   `f`, initial `t`, and safe `j/k/x` contexts. Reuse OCR only to locate candidates.
   Reject ambiguous source labels and intrusive neighboring ink; preserve words
   where single-letter separation is unsafe.
4. Add training-only near-duplicate and region-overlap grouping before any new
   independent split. Exact file uniqueness from this audit is insufficient.
   Keep known same-line, same-word, cross-DPI and transformed descendants together.
5. Turn the source-separated adapter proposal into a versioned dataset only with
   a frozen experiment protocol. Retain old Validation as exposed development,
   and report all reconstruction, train-text generation and novel-text generation
   separately.

None of these tasks presently requires the author to rewrite the alphabet or
relabel clear accepted samples. Author intervention is needed only for unresolved
original labels, a proven missing data class after the historical search, a
meaningful candidate's personal-style comparison, or use/promotion that the master
plan explicitly reserves for the user.

## Execution and verification

Actual command:

```bash
PYTHONPATH=.:src .venv/bin/python /tmp/hwsol-m3-corpus-readiness.py
```

The exact script is retained in the evidence run so the temporary path is not the
only reproduction source. It used the existing CPU environment and `verify_run`,
verified the original input run inventories, checked the original PDFs' byte hashes
without rendering protected full pages, compared all 158 selected RGB word crops
to their existing reviewed parents, checked local/mathematical source assets,
and rechecked protected hashes before sealing. It did not run model inference.
Run hardware and Python identity are in `run.json`; no GPU was required.

A preliminary read expected `reports/indices.json` in the v0.9 expanded word run;
that version stores splits in `samples.json`. The mistaken read was corrected
without changing any historical asset. No new test count or optimizer count is
used as evidence of handwriting quality.

**Go:** unified source registration, conservative historical candidate review,
and a pre-registered bounded context experiment after M0/M1 prerequisites.
**No-Go:** claiming a new independent personal holdout, full Corpus 2.0 completion,
validated natural joining, model publication, or starting RL from this audit.

## Executed federated registry, with incremental replay

The reusable importer is now implemented in
[`research/corpus_registry.py`](../research/corpus_registry.py). It uses the existing
immutable `Run` contract, takes repeatable explicit source manifests and supports
`--parent <registry-run>`. It does not change the SQLite catalogs or infer new
labels. The initial completed schema is `federated-handwriting-registry/2`.

Final initial registration:
[20261120T103956559524Z-federated-corpus-registry-097d0533](../workspace/runs/20261120T103956559524Z-federated-corpus-registry-097d0533/run.json).
Same-source incremental replay:
[20261120T103957791374Z-federated-corpus-registry-3096c754](../workspace/runs/20261120T103957791374Z-federated-corpus-registry-3096c754/run.json).
Both pass `verify_run` and remain `held`: registering evidence is not a training
admission or production release. Two earlier schema-1 initial/replay pairs remain
preserved as implementation trials; they are superseded by the above schema-2
runs, which also record review queries and group split checks.

| Registered entity | Actual count | Interpretation |
| --- | ---: | --- |
| Original documents | 11 | The eight English documents, one further historical math document and M/N. |
| Original region observations | 297 | 39 lines + 174 words + 84 math regions; these overlap hierarchically and must not be summed as independent training samples. |
| Distinct original words | 174 | `158 + 68 - 52`; the 18 strong words already belong to the 68-parent set. |
| Derivative/source views | 609 | 39 line, 174 original word, 18 normalized word, 84 math and 294 glyph views. |
| Declared training-target glyph views | 275 | The extra 19 glyph preprocessing views are lineage references, not new handwriting or new current targets. |
| Dataset/view memberships | 679 | Separate dataset roles and historical exposure, not independent observations. |
| Verified asset paths / distinct SHA-256 values | 1,224 / 714 | Includes path aliases and historic derivative images; no raw scan was changed. |
| Math source review | 76 accepted / 8 held or unresolved | Raw Agent review claims retained; no held candidate was promoted and no human review was invented. |
| New real handwriting / neural updates | 0 / 0 | Registration and provenance validation only. |

The replay added **zero** documents, observations, views, memberships, manifests
or asset paths. The initial parent run hash stayed
`86afa39fb363218257de5f92c67de6784e3a36a24da0713698ab27857c2dbb7e`;
formal `workspace/latest.json` was also checked unchanged. The larger verified
asset inventory compared with the earlier 667-hash audit includes explicit raw
and prior glyph preprocessing images and normalized whole-word targets; it is
not additional personal training data.

The importer snapshots each complete input JSON byte-for-byte under
`solutions/manifests/<sha>.json`; source references retain manifest identity, row
index, original review fields and word position where needed. Each version also
stores its executable importer and normalized `solutions/import-spec.json`.
Original document hashes, RGB source crops, source page, physical source rectangle,
parent coordinates, transforms, labels, capture/session groups, historic split and
exposure survive federation. Word and math crops are checked byte-for-byte in RGB
against their declared parent rectangle. This checks source-manifest consistency;
it does not substitute a new human transcription or re-render every original PDF.
Legacy word manifests without DPI require an explicit `source_dpi` in the import
spec; this import declares their audited 300-DPI source convention rather than
silently guessing it.

The registry rejects changed labels, source group/session identity, an existing
membership's split/exposure, missing assets, wrong asset/manifest hashes, corrupt
parent snapshots and sealed Session C. A new view remains a view of its original
region, and cannot increase the original-observation count merely by changing
preprocessing or DPI. Parent versions remain independently verifiable.

`reports/quality.json` makes review and leakage queryable. For every dataset it
lists source group, source document and available capture-session membership by
split. Different views/words in one source group across Train/Validation produce
`group_split_gate: no-go` and a held version; the importer preserves both historical
memberships instead of silently changing their split or dropping them. The actual
registered subset has no such conflict. This is **not** evidence of a clean unseen
system holdout: the eight English sources remain exposed to the frozen backbone,
and the larger 613-view dataset's old Validation was not reclassified or imported
as a new final test. Every view retains `training_ready: false`; source label review
and the task-specific training admission decision remain separate. Unknown or held
source review is never converted into acceptance based on the expected label.

[Unified registry](../workspace/runs/20261120T103956559524Z-federated-corpus-registry-097d0533/reports/registry.json)
· [review/group split queries](../workspace/runs/20261120T103956559524Z-federated-corpus-registry-097d0533/reports/quality.json)
· [replay result](../workspace/runs/20261120T103957791374Z-federated-corpus-registry-3096c754/reports/result.json).

Reproduction creates another fresh private run:

```bash
.venv/bin/python -m research.corpus_registry \
  --spec workspace/runs/20261120T103956559524Z-federated-corpus-registry-097d0533/solutions/import-spec.json \
  --parent workspace/runs/20261120T103956559524Z-federated-corpus-registry-097d0533
```

For future incremental data, a spec supplies a `manifests` list with `kind`, `path`,
`dataset_id`, historical `split`, an explicit `exposure` list, optional expected
`sha256`, and declared `source_dpi` if the old record lacks it. Supported adapters
are `documents`, `lines`, `words`, `glyphs` and `math`, using the retained source
schemas. Repeated `--manifest KIND=PATH` is also available when no per-source
metadata override is needed. A new source is still subject to its own label and
source review; this interface performs no OCR auto-approval.

Executed focused checks:

```bash
.venv/bin/ruff check research/corpus_registry.py tests/test_corpus_registry.py
.venv/bin/python -m pytest -q tests/test_corpus_registry.py
```

The checks exercise immutable replay, retained raw records/transforms, derivative
identity, label/group/split conflict, missing/hash-mismatched assets, parent-pixel
mismatch, sealed-source refusal, parent tampering, held math review, and cross-view
source-group split leakage. They establish registry behavior, not handwriting
quality. No training was run. This completes the actionable registration increment;
the full Corpus 2.0 Studio, independent author data acquisition, continuous-ink
review and M4 A/B/C task-specific admission remain the work described above.
