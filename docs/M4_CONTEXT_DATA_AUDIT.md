# M4 historical context supervision audit

2026-11-20. **Continuous personal source material is available for development;
the currently proposed local A/B/C training protocol remains No-Go.** The blocker
is supervision alignment and separation of geometry, source style and contextual
effects. It is an engineering/data-preparation gap, not absence of handwriting or
a reason to ask the author to do Agent annotation work.

The fixed review found 28 words with clear cross-letter ink, 11 with naturally
separate strokes in the reviewed spans, and one held for insufficient fine detail.
Those 28 positives are outside the old 13-word adapter training set. They include
shared crossbars and natural start/end-stroke relationships that the original
18-word preflight did not cover. Disconnected writing remains valid personal
style and useful whole-word supervision; connection is not the definition of
context or a quality objective by itself.

Final sealed evidence:
[20261120T110917401810Z-context-supervision-audit-cacf3188](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/run.json).
[40-candidate RGB contact PDF](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/comparisons/candidates/contact.pdf)
· [five-page enlarged review PDF](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/comparisons/candidates/detail/contact.pdf)
· [all 174 original words](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/comparisons/words/contact.pdf)
· [all 39 original lines](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/comparisons/lines/contact.pdf)
· [explicit per-candidate review](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/reports/visual-review.json).

## Scope, source identity and counting

The input is the immutable M3 federated registry, not new OCR output. It contains
174 original word regions and 39 accepted line regions from eight source groups.
The words and lines overlap and are not additive independent samples. All eight
English source documents were exposed upstream to the frozen v21 backbone.
A/B are known capture sessions; the other historical PDF groups do not establish
different writing dates or independent sessions.

Every inventory entry retains original label, source PDF SHA, page, original
region coordinates, crop hash, parent geometry, full raw source records and prior
review fields. The raw RGB crops are copied byte-for-byte. String-index locations
identify requested character/sequence positions, but do not invent per-character
pixel boxes. Original labels, originals, Profiles, checkpoints, historical splits
and formal latest remain unchanged. All adjacent-letter pairs are indexed by string position as well as the requested
f/i/t and sequence families; spaces/punctuation do not create artificial pairs.
New review notes are an append-only audit,
not new transcriptions or author confirmation.

| Source group | Original words | th | st | ll | ing | ff | fi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Session A | 48 | 12 | 1 | 1 | 0 | 1 | 1 |
| Session B | 28 | 8 | 4 | 2 | 1 | 1 | 1 |
| Physics 2 | 45 | 7 | 1 | 0 | 0 | 0 | 1 |
| Calculus 1 | 28 | 4 | 0 | 0 | 1 | 0 | 0 |
| Calculus 2 | 10 | 0 | 0 | 0 | 0 | 0 | 1 |
| Calculus 3 | 4 | 0 | 0 | 0 | 0 | 0 | 2 |
| ICS | 7 | 0 | 0 | 1 | 0 | 0 | 1 |
| Formula Sheet I | 4 | 0 | 1 | 0 | 0 | 0 | 0 |
| Total | 174 | 31 | 7 | 4 | 2 | 2 | 7 |

There are also seven `ti` occurrences and no `ft` in the registered word labels.
These are **textual opportunities**, not connected-stroke counts. Case folding
is used only for analysis; source labels keep their original capitalization.

| Character, ignoring case for this table | Word-initial | Word-medial | Word-final |
| --- | ---: | ---: | ---: |
| f | 10 | 7 | 1 |
| i | 14 | 58 | 0 |
| t | 34 | 45 | 19 |

The exact case-sensitive totals are `f=17, F=1, i=71, I=1, t=88, T=10`.
The final `f` occurs in intact `of`; its crossbar meets the preceding o, so it is
not automatically a safe new isolated f target. This does not increase the old
local model's three accepted f crops.

The 39-line inventory separately contains 32 th, 6 st, 4 ll, 2 ing, 3 ff, 4 fi,
0 ft and 22 ti textual occurrences. Its f/i/t positions are **relative to the
entire line**, not to individual words. Final records use `region_position` and
explicit `position_scope: within-line` or `within-word`. The earlier preparation
run `20261120T105739621461Z-context-supervision-audit-e1a07d36` is preserved held:
its line-relative values used a misleading `word_position` field name. The corrected
run `20261120T110354584581Z-context-supervision-audit-6d277408` fixes this naming.
The final replay additionally indexes all adjacent-letter pairs and retains that
run unchanged. Its 40 candidate IDs, labels and crop hashes were checked identical
before reusing the explicit visual review; this is not another 40 new samples.

[Full coverage](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/reports/coverage.json)
· [all source records and coordinates](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/reports/inventory.json).

## Bounded visual review

Selection was fixed from labels and source identity before visual inspection:
all f-containing words and st/ll/ing candidates, then distinct source-group/text
th examples, then one `it` control. Forty words across all eight groups were
reviewed; only two are from the old 13-word set, and both were negative for the
inspected connections. There was no search expansion after seeing images.
All 40 were inspected in RGB contact sheets and enlarged sheets; selected small
examples were inspected again at pixel-preserving magnification. No connected-
component threshold or intended text automatically declared a join.

| Reviewed span | Clear cross-letter ink | Separate strokes | Held |
| --- | ---: | ---: | ---: |
| ff | 2 | 0 | 0 |
| fo | 6 | 1 | 0 |
| th | 9 | 5 | 0 |
| st | 0 | 7 | 0 |
| ll | 0 | 4 | 0 |
| ing | 0 | 1 | 1 |
| fi | 1 | 6 | 0 |

The review also records visible te, ta, co, he and other relationships when they
are clear within a selected word. Counts are local span decisions, not independent
new word samples. A word with connected fo and separate ll appears in both
relevant span rows but only once in the 28/11/1 word-level result.

Concrete positive evidence includes the shared ff crossbar in `coefficient`
(Session A) and `sufficiently` (Session B); fo crossbar contacts in multiple
original sources; and th return/ascender relationships in `throughout`, `these`,
`that` and natural Physics words. The same author also writes separated th, ll,
st and fi. `for` occurs both with and without f/o contact in the same ICS source
page: those examples are useful variation controls and must remain in the same
source group when splitting data.

The held `according` source retains its existing full-word label. Its native
resolution is insufficient for confidently specifying the fine ing connection
geometry. It is not repaired, relabeled, or used as a positive junction target.
No confirmed cropped-off character was silently accepted. RGB grid/ruling and pen
color remain visible so that graph lines are not counted as handwriting joins.
Offline contact does not prove pen trajectory or stroke order.

The 28 positive words cover seven source groups; seven observed pair families
occur in multiple groups. The preflight's illustrative 20-word/3-group reference
is therefore met. **That reference is not an absolute scientific threshold and
does not automatically admit training.**

## Available supervision versus unresolved experiment design

Intact source words and lines already supervise word proportions, shared slant,
height rhythm, pen character, start/end relationships and natural variability,
including examples with no connections. Fourteen case-folded text groups occur
in multiple source groups; e.g. `for`, `the`, `this`, `constant`, `it` and
`approximately`. Exact case must remain separate where it affects glyph identity.
[Repeated-content inventory](../workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188/reports/repeated-content.json).

This is enough genuine material to justify contextual development and a new
alignment feasibility comparison. It is materially different from rerunning the
old 13 clean reconstructions. However, the proposed **local** A/B/C experiment
needs a trustworthy correspondence between its supervised interaction regions
and original source ink. Most newly reviewed words have accepted whole-word and
parent-line coordinates, not reviewed semantic pair-region or per-letter geometry.
The original 18-word boxes must not be silently extended to them using guessed
cuts. Word crop aspect ratio alone is also not a verified shared slant/x-height
annotation.

A naive whole-word loss can improve placement, duplicate crossbars, scan/pen
contrast or background handling while being described as a context benefit.
Sources range from fine/red writing to heavy black writing. Those source effects
and the model's sampling/conditioning gap need matched controls. This audit has
not yet generated that fixed-alignment A/B comparison or established the C mask.
Its task-level result is therefore **No-Go for the current, not-yet-frozen training
protocol**, while original context data readiness is positive. No author action
is needed to resolve this engineering gap.

## Next executable minimum alignment task

Use the following **already reviewed** 12 positive spans and four natural
unconnected controls. This is a bounded alignment pilot, not a declaration that
16 is enough for M4 style acceptance. Source rectangles below are the complete
word region in the original 300-DPI page raster, top-left origin,
`[left, top, right, bottom]`. The listed string spans are zero-based half-open
indices. Pair pixel rectangles remain **unannotated**, deliberately: the next
Agent must mark them from the original word without inventing individual glyph
boundaries. Exact PDF transforms must come from the retained source provenance,
not a new nominal-DPI conversion that ignores rotation or page crop.

| Word ID / unchanged label | Span | Source SHA prefix / page | Original whole-word rectangle | Role |
| --- | --- | --- | --- | --- |
| w017 coefficient | ff [3,5) | 3f3c6131 / 1 | [451,1676,790,1769] | positive, Session A |
| w070 sufficiently | ff [2,4) | 5314eb36 / 1 | [1200,1688,1542,1775] | positive, Session B |
| w012 follows | fo [0,2) | 3f3c6131 / 1 | [605,2211,826,2308] | positive, Session A |
| w063 following | fo [0,2) | 5314eb36 / 1 | [447,1420,741,1505] | positive, Session B |
| w168 for | fo [0,2) | bd60445b / 95 | [932,1000,1007,1081] | positive, Physics |
| w170 for | fo [0,2) | 4db1f459 / 7 | [584,1331,680,1417] | positive, ICS |
| w020 throughout | th [0,2) | 3f3c6131 / 1 | [1481,1678,1816,1775] | positive, Session A |
| w060 these | th [0,2) | 5314eb36 / 1 | [1244,1143,1444,1205] | positive, Session B |
| w006 the | th [0,2) | bd60445b / 59 | [526,912,690,1046] | positive, Physics |
| w093 that | th [0,2) | 77f119da / 21 | [558,249,730,302] | positive, Calculus 1 |
| w101 they | th [0,2) | bd60445b / 59 | [579,1294,697,1366] | positive, Physics |
| w118 the | th [0,2) | 77f119da / 21 | [830,1047,934,1088] | positive, Calculus 1 |
| w113 for | fo [0,2) | 4db1f459 / 7 | [700,1440,796,1530] | natural unconnected control |
| w013 the | th [0,2) | 3f3c6131 / 1 | [877,2220,975,2288] | natural unconnected control |
| w045 with | th [2,4) | 3f3c6131 / 1 | [1188,1951,1325,2027] | natural unconnected control |
| w076 both | th [2,4) | 5314eb36 / 1 | [1420,864,1548,935] | natural unconnected control |

Original files are `solutions/words/<word-id>-original.png` in the final audit
run; full paths, document hashes, parent crops and transforms are in
`reports/inventory.json`. Word labels/regions have existing source evidence and
current visual consistency checks. Connection classifications are explicit Agent
judgments; pair-coordinate confidence remains **pending** until the next task.

The minimum work is:

1. Add 16 Agent-reviewed **intact span** rectangles and context masks in new
   evidence, retaining entire words and parent lines. Record word-relative and
   page-raster coordinates, original RGB SHA, inherited transforms, semantic
   string indices, reviewer, geometric confidence and ambiguity. Include complete
   ascenders, descenders, dots and shared bars. Do not cut ff or th into artificial
   independent characters. If a span cannot be separated from neighbors safely,
   retain whole-word supervision and mark the local mask unavailable.
2. Recheck those 16 annotations once against source pixels and surrounding context.
   Keep all derivatives of each original PDF/session together. ff has only two
   words across A/B, so it cannot alone support a robust style/generalization claim.
3. Render A and a zero-state B for the 16 unchanged contents with fixed model,
   Profile, seed and layout: about **32 small word outputs**, no parameter updates.
   B's zero-state must recover A. Save pre-alignment output as well as diagnostics.
   Apply one declared word-level transform policy equally; do not let C improve
   a target-specific warp while A is held unaligned. Use a small predefined
   transform sensitivity check to identify alignment-dominated cases.
4. Compare ink in the reviewed intact spans, shared word proportions/slant and
   the unconnected controls. Retain true-condition reconstruction separately from
   text-only generation. If a stable contextual residual cannot be distinguished
   from alignment, thickness or source effects, stop this proposed local protocol.
   If it can, freeze one bounded A/B/C protocol and its source-group development
   split before any training. Independent author/style acceptance remains later.

The first 16 annotation/recheck records and 32 zero-update outputs are the minimum
new workload; no additional document search, character collection or full alphabet
sheet is required for that engineering pilot. All 39 unheld reviewed candidate
words can remain intact continuous targets without forced single-character cuts;
the one low-detail target is held specifically for fine junction supervision.

No new author writing is requested now. If later frozen-candidate acceptance needs
unexposed author references, request ordinary short phrases at the author's natural
pace, with both naturally joined and naturally separated behavior allowed. Do not
ask the author to manufacture a specified ligature or to solve our alignment task.

## Execution and verification

Source:
[`research/context_supervision_audit.py`](../research/context_supervision_audit.py).
The prepare command records inventories and contact sheets; finish requires an
explicit per-candidate review plus a separate task assessment. Counts never assign
a training Go. The exact submitted review and source snapshots are stored in the
run. The CLI and equivalent Python `prepare`/`finish` API were executed; the
final replay checks candidate/hash identity before reusing the saved audit.
The reproduction entry points are:

```bash
.venv/bin/ruff check research/context_supervision_audit.py
.venv/bin/python -m research.context_supervision_audit prepare
.venv/bin/python -m research.context_supervision_audit finish \
  workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188 \
  /tmp/hwsol-context-visual-review.json
```

The temporary review input is preserved as `reports/submitted-review.json` and its
normalized record as `reports/visual-review.json`. A reproduction must create a
fresh run; finalized runs cannot be finished again. `verify_run` passes. All source
hashes and formal latest stayed unchanged. Zero network inference, zero neural
updates, no sealed C access, no source label changes and no new collection sheet.

The bounded audit is complete and sealed. The main Agent owns the subsequent
Go/No-Go and commit decision; this audit neither starts training nor expands the
search beyond the registered sources.
