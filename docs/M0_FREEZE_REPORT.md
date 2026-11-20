# M0 — actual baseline freeze and reproduction

2026-11-20, branch `feature/master-execution-m0-m1`.

The user's complete instruction is preserved in [MASTER_EXECUTION_PLAN.md](MASTER_EXECUTION_PLAN.md),
SHA-256 `150e60e9f28adbc343507e4aa9715d3057200b68ba06a2446caa8a4e94445bff`.
The authoritative entry is [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Frozen combination

| Component | Actual reference |
| --- | --- |
| Production | main `b2754e7`, release v0.6.0 `2bcb260`; CPU Handnote/MathText |
| Default / formal latest | personal@v1 / v06-five-final using personal@v3; unchanged |
| English generator | canonical-conditioned-ink/1, 194,513 parameters, 96 × 96 local ink |
| Checkpoint | canonical-ink-component-refinement-fbea19c9, checkpoint-01000.pt |
| Weight SHA-256 | `779a2ab51902a17a7068df2c32bf7f5ceddef938ee6982f116f39e0dfb3ad15d` |
| Real local data | 275 views, 28 real character classes; 44 other classes remain font-assisted |
| English layout | v22 metrics, v23 measured word rhythm; 477 measured gaps |
| Mathematics | v26 D resolver, original historical/M/N contours and calibrated structures |

The model has 6,000 updates in its lineage, including the last run's 1,000.
These numbers identify the model; they are not evidence of handwriting quality.
Current sampling uses class-local observed PCA conditions with bounded noise.
This is a local ink generator composed into words, not a trained continuous-word model.

## Actual reproduction

- [Freeze run](../workspace/runs/20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e/run.json):
  58 protected assets unchanged, existing production PDF/PNG verification passed.
- [Independent C/D replay](../workspace/runs/20261120T102430665658Z-v01026-session-math-polish-7828d221/run.json):
  all 14 page images exactly match the archived comparison.
- [Replayed seven-page D PDF](../workspace/runs/20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e/rendered/homework_solutions.pdf),
  [page PNGs](../workspace/runs/20261120T102428732053Z-m0-baseline-freeze-ea6ffa5e/rendered/pages).
- [Actual neural re-inference](../workspace/runs/20261120T103322401823Z-m0-neural-ink-replay-d65c7450/run.json):
  539 input records reproduced; 529 generated word PNGs, floating-point ink arrays
  and condition arrays match exactly. This upstream inventory differs from the
  526 neural words actually placed in the final first four pages.

The remaining three D pages use CPU English. This reproduction does not establish
a universal fresh-text neural backend, independent style acceptance, or complete
mathematical symbol coverage. No new training occurred.

Commands executed from this checkout:

```sh
PYTHONPATH=.:src .venv/bin/python -m research.project_state freeze
PYTHONPATH=.:src .venv/bin/python -m research.project_state verify
PYTHONPATH=.:src .venv/bin/python -m research.project_state replay-neural
```

Each run contains the actual worker source/request and logs. Neural inference uses
the existing isolated `workspace/environments/english-htr-v08` environment; GPU
access required sandbox escalation. Hardware is NVIDIA RTX 5060 Laptop, 8,151 MiB,
driver 580.126.09. The CPU registry/replay path uses Python 3.12.16 without importing
Torch. Existing checkpoints and old run directories were not written by replay.

The private production/research/candidate registry entries point to the sealed
freeze receipt and are verified against it. `latest.json` remains SHA-256
`b237f5cc98854ae04e2b9ffd34e6ab111233fba77a66a66cabf32c6c6bc1132a`.

Decision: **Go for M1**. **No production promotion**. Reproduction establishes an
auditable baseline, not a new content or personal-style result.
