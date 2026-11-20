# M1 complete-homework benchmark protocol, version 1

Status: protocol prepared before any M1 rendering. This file and `research/homework_benchmark/cases.py` must be hashed in the evaluation run before it starts. No render outcome has been used to select or simplify these cases.

The question is whether the frozen production CPU path and the frozen v0.10.26 personal-math candidate handle new complete homework content, preserve mathematical structure, and report unsupported notation and nonpersonal ink accurately. Both arms use CPU English on these new problems. Cached English images from the old four pages cannot demonstrate new-content generation and are not used here. This evaluation does not train a model or claim independent author-style generalization.

## Inventory and source status

The corpus contains 24 newly Agent-authored university-level problem statements with complete concise English reference solutions. There are six problems each in linear algebra, calculus, statistics and physics. They were authored for this benchmark on 2026-11-20, are not copied historical university assignments, and have no paired unseen personal handwriting. Their source kind and author are explicit in every record. Reference solutions are benchmark assets, never personal training data.

Sixteen cases are development (four per domain); eight are one-use evaluation holdout (two per domain). The holdout means unseen by renderer tuning for this version, not unseen by the question author or a claim of new writer data. Mathematical reference verification may occur before rendering, but renderer adjustments must not use holdout outcomes. All older four/seven-page outputs and benchmark-v1 cases remain regression, with their original exposure status. Original handwritten Session C is not opened or used.

Twenty-one cases are intended to produce one page. Three development cases deliberately preserve currently unsupported notation: a native piecewise `cases` environment, a probability `underbrace`, and quantum expressions containing `hbar`. The expected outcome is an explicit refusal with the recorded diagnostic. They are complete mathematical problems, not blank or malformed inputs. Their source cannot be silently rewritten to make them pass. Any later equivalent-notation support must be a distinct development revision preserving the old failure and source.

The set includes RREF/null spaces, matrix powers and products, complex inner products, double integrals, Taylor bounds, improper integrals, weighted regression, likelihood products, chi-square and paired tests, vector fields, closed-loop integrals, RLC damping, wave functions and optics. New symbols are to be measured against the old seven-page provenance, not presumed personal simply because a command parses. Long explanatory prose, repeated letters, several subparts, dense derivations, and grid/dot/custom-image paper supplement the default white A4 cases.

## Freeze and execution order

1. Bind production Profile/source identity and v0.10.26 personal-math/Profile identity to the M0 freeze receipt. Record the benchmark source and this protocol SHA-256.
2. Run exact symbolic/numerical reference checks. Save their results separately from renderer checks. Independently review the solution derivations before declaring the references verified.
3. Materialize development jobs in new run directories. Render both arms with matching page dimensions, content, seed and background. Preserve all successes and failures.
4. Diagnose only development failures. If a change is needed, create a new source identity and new run; never overwrite old outputs. Recheck frozen regression outputs where the change can affect them.
5. Freeze the final candidate and quality criteria before opening the eight holdout rendering results. Execute one complete holdout pass; record its consumption and candidate identity. An exact reproducibility replay is allowed but is not a second independent test.
6. If a holdout defect leads to a fix, those cases become regression. Any new independent claim requires separately authored and frozen evaluation content. Do not relabel or replace failed holdout examples.

The runner must reject a holdout invocation without a freeze receipt. A local consumption ledger should bind corpus version, all case hashes, candidate hash, first evaluation run and timestamp. A failed infrastructure invocation is retained and may be resumed with its unchanged identity; inspecting a completed output consumes that case.

## Predeclared quality decisions

| Dimension | Requirement and interpretation |
| --- | --- |
| Reference content | All executable exact checks pass. Remaining analytical claims receive an explicit Agent/human review status. Correct numeric spot checks alone do not prove the derivation. |
| Supported rendering | Exactly one A4 page per top-level problem, all subparts retained, no known omitted/changed text, formula, unit or sign. A render failure is reported as a capability failure, not hidden by removing the case. |
| Unsupported rendering | Refuse with the expected unsupported-command/environment diagnostic and retain the full input. No accepted PDF that silently drops the notation. |
| Mathematical structure | Record source LaTeX, parser success/refusal, normalized source representation, matrix dimensions and structural roles where exposed, item identities/order, and before/after math trace. Source normalization is not a full semantic AST or proof of rendered equivalence. |
| Glyph provenance | Every visible mathematical item needs its actual source category, font/outline identity, collected/derived/programmatic status, fallback and reason. There is no blanket zero-fallback requirement for unseen symbols; unreported fallback fails. |
| Ink and page limits | No confirmed clipping, page overflow, empty expected glyph or newly introduced content collision. Bounds checks and independent visual review are both recorded. Unknown visual cases remain review-required. |
| English | Both arms retain the same CPU text and layout inputs. The candidate cannot claim neural generalization from old cached word images. Missing textual content is a failure regardless of mathematical appearance. |
| Backgrounds | Same selected background in both arms, correctly placed below ink. Background marks are not counted as solution ink. |
| Reproducibility | Record source, data, Profile, configuration, seeds, run and output hashes. Frozen output pixel replay proves reproduction, not content or author similarity. |
| Release | Any known content/math defect in an accepted output prevents that candidate's release. All unresolved cases and unsupported boundaries must be explicit. M1 engineering success does not substitute for M2 author visual approval or cross-platform evidence. |

No aggregate authenticity percentage or training/test-count quality score is used. Counts of supported/refused/review-required cases are reported with denominators. No author preference win threshold is retrofitted to these render-only cases. Independent style evaluation belongs to the later preregistered comparison with the strongest usable baseline.

## Separate end-to-end exercise

Four additional Agent-authored original plain-text inputs are provided by `end_to_end_inputs()` with one question per domain. They deliberately have no reference solution in this module. A different execution step must read each original, inventory subparts, solve, independently check, invoke the Skill/runtime and inspect the delivered page. Save problem input, transcription, solution derivation, verification evidence, actual commands and source inventory. This is a text-input end-to-end exercise; do not claim image/PDF recognition until original image/PDF inputs have actually been read. Do not copy render-only answers and rename that run end-to-end.

Errors are assigned to intake/transcription, solving, verification, layout, glyph generation or export. An accurate rendering of an incorrect solution is an Agent failure, while a missing sign from a correct source is a renderer failure. Both remain visible in the final report.

## Reuse interface

`cases(split)` returns plain dictionaries. Each has `id`, `split`, `domain`, `title`, `statement`, `solution`, `source`, `expected`, `background`, `density`, `numeric_checks` and `semantic_checks`. `expected.outcome` is `render` or `reject`; `expected.error_contains` describes deliberate refusals. `verify_references()` executes exact SymPy checks and returns per-case evidence. `write_inputs(dest, split, profile)` writes existing Job-schema manifests plus original statement, editable solution and required local paper assets into new case directories. It never renders, mutates a Profile, updates latest or trains a model.
