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
| Math research baseline | v0.10.26 D, original personal glyphs + measured structures + spacing/stem correction | Seven-page evidence; no general coverage claim |
| Current milestone | M0 complete; M1 new-content benchmark and M3 corpus audit | In progress |

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
from public source and release archives. No production activation is authorized.

M0 verified freeze run: [20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e](../workspace/runs/20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e/run.json).
The C/D replay has 14 identical page images and 58 unchanged protected assets.
[Replayed D PDF](../workspace/runs/20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e/rendered/homework_solutions.pdf).

M0 also re-ran the actual frozen checkpoint: 529 word PNG/float/condition arrays exactly match.
See [M0 evidence and limitations](M0_FREEZE_REPORT.md). No new neural training occurred.
