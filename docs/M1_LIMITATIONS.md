# M1 known limitations and explicit exposed-case recovery

2026-11-20. The original one-use evaluation remains **7 rendered, 1 rejected out of
8**. Its failed case, source, protocol and consumption ledger are unchanged.
This document records an Agent-level equivalent source revision after exposure;
it does not turn the original result into 8/8 or add renderer capability.

## Cross-matrix scalable delimiters

The original failure is `m1-la-06-rank-compatibility` in
`20261120T104341625106Z-homework-m1-render-38ea743e`. The exact solution SHA-256 is
`f3e9e72dd0c3cabdc12c929f9f7d638714ea3943eb885c7ce0f4fa9d927011e8`.
It contains the valid mathematical expression:

```latex
\mathrm{Null}(A)=\left\{t\begin{pmatrix}-2\\1\\1\end{pmatrix}:t\in\mathbb{R}\right\}
```

The pinned Handnote `_display_blocks` in `handnote/markdown.py` retains a formula
with material after a matrix as a display formula. `Renderer._compose_math` in
`handnote/render.py` then splits it at `MATRIX_RE` matches, rendering each flat
piece independently. Its first piece is `\mathrm{Null}(A)=\left\{t`, which has
no `\right`; the final piece has `\right\}` without its matching `\left`.
MathText rejects the incomplete group. Upstream catches this and attempts plain
formula fallback; `SinglePageRenderer.math_image` in `src/hwsol/jobs/layout.py`
correctly rejects that unauditable fallback during width planning. No accepted
PDF silently omits the matrix. The personal-glyph Resolver has not run yet.

Read-only in-memory controls confirmed that the isolated column matrix, the
complete set expression without a matrix, and a flat matrix with independent
trailing math each parse; the group spanning a matrix fails. This is a grouping
and layout limitation, not missing brace collection or insufficient training.

A future generic fix would require balanced, escape-aware delimiter groups and
layout of their child matrix row before sizing both sides. Baselines, invisible
delimiters, nested groups, item provenance and the existing matrix-stroke role
inference need separate regression. Matrix-containing fractions or radicals
would remain unsupported unless explicitly implemented. No such change was
made for this recovery; the current refusal remains the supported behavior.

## One explicit equivalent revision

[`recover_exposed.py`](../research/homework_benchmark/recover_exposed.py) requires
the exact archived solution hash and exactly one occurrence of the failed
formula. Revision `homework-m1-v1-exposed-recovery-1` replaces only that source
span with:

```latex
v=\begin{pmatrix}-2\\1\\1\end{pmatrix}

\mathrm{Null}(A)=\left\{tv:t\in\mathbb{R}\right\}
```

The script checks that every character outside that span is unchanged. It saves
both sources, the original rejection, a unified diff, the original Job, the new
Job, Profile snapshot and source hashes. The new Job explicitly requests the
actual frozen comparison mode `baseline` and pins its Profile manifest; the
archived template's `combined` setting is retained only in the original Job.
This is not an automatic renderer rewrite and never reads sealed Session C.

Independent exact SymPy checks, separate from renderer traces, verify `A=BC`,
independence of the two columns, rank two, the left-null compatibility condition,
its sufficiency, the particular solution for `(3,-2,4)`, the complete nullspace
basis `(-2,1,1)`, and all real multiples. Naming that same column `v` preserves
the parameter domain, coordinate order and complete solution set. These
algebraic checks alone do not verify visible handwriting.

Reproduce in a new run, without writing latest:

```bash
PYTHONPATH=.:src MPLCONFIGDIR=/tmp/hwsol-m1-recovery-mpl .venv/bin/python -m research.homework_benchmark.recover_exposed
```

## Actual output and Agent visual review

Recovery evidence run:
[`20261120T104852855240Z-m1-exposed-recovery-df8c7f2b`](../workspace/runs/20261120T104852855240Z-m1-exposed-recovery-df8c7f2b/run.json).
The unchanged public CPU `render_job` path was also exercised with an isolated
copy of the pinned Profile and passed `verify_job(pixels=True)`:
[CPU PDF](../workspace/runs/20261120T104852855240Z-m1-exposed-recovery-df8c7f2b/rendered/production-cpu/homework_solutions.pdf),
[CPU PNG](../workspace/runs/20261120T104852855240Z-m1-exposed-recovery-df8c7f2b/rendered/production-cpu/pages/01.png),
[source diff](../workspace/runs/20261120T104852855240Z-m1-exposed-recovery-df8c7f2b/reports/one-formula-change.diff),
[exact algebra](../workspace/runs/20261120T104852855240Z-m1-exposed-recovery-df8c7f2b/reports/semantic-checks.json).

The associated isolated comparison is
[`20261120T104853010542Z-homework-m1-render-7595af17`](../workspace/runs/20261120T104853010542Z-homework-m1-render-7595af17/run.json):
[complete A/B PDF](../workspace/runs/20261120T104853010542Z-homework-m1-render-7595af17/comparisons/A-B-homework.pdf),
[candidate PDF](../workspace/runs/20261120T104853010542Z-homework-m1-render-7595af17/rendered/homework_solutions.pdf),
[candidate PNG](../workspace/runs/20261120T104853010542Z-homework-m1-render-7595af17/rendered/pages/candidate/job-001.png),
[all formula comparisons](../workspace/runs/20261120T104853010542Z-homework-m1-render-7595af17/comparisons/formula-regions.pdf),
[provenance](../workspace/runs/20261120T104853010542Z-homework-m1-render-7595af17/reports/job-001-provenance.json).

The benchmark-runner Agent inspected the complete paired page, the complete
formal CPU PNG and all 10 native-resolution formula/matrix pairs. The observed
matrix entries and signs, compatibility relation, particular-solution column,
real parameter domain, defined nullspace column and final set expression match
the revised source. No omitted subpart, clipped ink or overlapping expression
was observed. The one-page composition fits comfortably; the standalone column
uses more vertical space than the former inline arrangement would have.

Engineering checks found zero changed English pixels across 396 text items and
zero changes outside recorded math slots. Font fallback occurrences are 7 in A
and 4 in B; the remaining membership/real-number glyphs are explicit fallbacks,
not claimed personal ink. B's small personal braces are thin and angular and
still need author style judgment. The unchanged original tall parenthesis style
also remains visible around the separate column. This review is not an author
blind preference test or evidence of general neural generation.

Both new runs remain held. Source-preservation and actual CPU rendering are
verified; production publication and author visual approval are not granted.
There were zero neural updates. The original eight-case result and formal
latest pointer were checked unchanged after completion.

## Historical Job templates versus actual execution settings

The earlier built-in runs materialized `cases.write_inputs()` authoring Jobs
with `realism=combined` and template render defaults. The built-in comparison
actually executed its archived `context_composer` source with `baseline`
configuration and the frozen `run.json.config.render_config`. Thus those
earlier `solutions/inputs/*/job.json` files are **authoring templates, not exact
execution or replay manifests**. The actual archived source, Profile snapshots,
case sources and run configuration are the execution evidence. Existing
archives are left unchanged; no provenance discrepancy is retroactively erased.

For future built-in runs, `render.bind_execution_jobs()` first preserves each
template byte-for-byte as `job-template.not-executed.json`, then exports the
actual `baseline` mode, frozen render configuration and exact Profile manifest
pin to `job.json`. `reports/execution-jobs.json` links both files and their
hashes and explicitly marks the template as not executed. This changes neither
the frozen case module nor its protocol. The resulting execution Job is checked
through the same `manifest_cases()` contract used by `--manifest`.

Diagnostic run
[`20261120T105256950708Z-m1-execution-manifest-check-59e89704`](../workspace/runs/20261120T105256950708Z-m1-execution-manifest-check-59e89704/run.json)
bound and validated the 16 development manifests, preserving all 16 original
template hashes. It rendered only the already-exposed first development case,
`m1-la-01-rref-null`, through the exported execution Job. Its
[comparison run](../workspace/runs/20261120T105257633255Z-homework-m1-render-0107abcb/run.json)
produced an A and B page with **zero changed pixels in each arm** against the
original development output. This verifies configuration/export consistency;
it is not a new independent benchmark or mathematical-quality claim. The
consumed holdout and its ledger were not rerun or changed.

The complete executable diagnostic is archived at
[`logs/execution-manifest-check.py`](../workspace/runs/20261120T105256950708Z-m1-execution-manifest-check-59e89704/logs/execution-manifest-check.py),
with its renderer source snapshot and protected-asset hashes. A new exposed
replay can also use the exact exported execution Job directly:

```bash
PYTHONPATH=.:src MPLCONFIGDIR=/tmp/hwsol-m1-execution-mpl .venv/bin/python -m research.homework_benchmark.render --manifest workspace/runs/20261120T105256950708Z-m1-execution-manifest-check-59e89704/solutions/inputs/m1-la-01-rref-null/job.json
```

This command creates a fresh run. It does not invoke `--split holdout`, replace
historical outputs or change the original 7/8 holdout rendering outcome.
