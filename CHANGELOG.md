# Changelog

## 0.3.0 · 2026-11-18

- Added pixel verification between each delivered PNG and its PDF page, and regression coverage for nested equations, rotated paper sizes and seeded real glyph alternates.
- Reused repeated outline measurements, checked white-paper visibility directly on the export, and isolated candidate font handles during Studio evaluation.
- Added CI for two Python versions on three operating systems. Added a fresh-environment wheel smoke test and packaging from a Git tag.
- Added a local research CLI with pinned DiffBrush and One-DM sampling adapters, DiffusionPen output import, blinded English transcription and style scoring. Production continues to use deterministic math.
- Linux test results, outstanding platform checks and the scope of the model experiments are recorded in the [implementation status](docs/IMPLEMENTATION_STATUS.md).

## 0.2.0 · imported baseline

- Agent Skill, one-problem-per-page Runtime and CPU Profile Studio, including versioned publication, replay, verification and editable solutions.
- Prior engineering evidence for Phases 0–2 is included in docs/evidence.
