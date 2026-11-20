# Repository execution guidance

Read `docs/MASTER_EXECUTION_PLAN.md`, `docs/PROJECT_STATUS.md`, and
`docs/EXECUTION_HANDOFF.md` before continuing this project. The first file preserves
the user's long-term instruction verbatim; the latter two record actual progress.
Follow newer user instructions when they supersede that plan.

Continue the existing repository and pinned Handnote implementation. Keep research
on an isolated branch and experiments in fresh `workspace/runs/<run_id>/` directories.
Preserve production `main`, `workspace/latest.json`, original scans, historical
labels, Profiles, checkpoints and sealed evaluation data. Do not add mandatory
Torch to the CPU runtime. Never count font calibration as neural training.

Proceed autonomously with authorized reversible development and evaluation. The
user permits parallel agents for non-conflicting work; the primary agent must
review their evidence and integrate results. Preserve failed trials and distinguish
training reconstruction, independent generation, engineering checks and author
style assessment. Do not tune on consumed holdout while calling it unseen.

Production activation/publication, external sharing of private data, destructive
asset changes and changes to a frozen final evaluation protocol require the user's
explicit approval. Prepare reviewable artifacts first. Continue independent work
while a genuinely user-dependent decision remains pending.

Update `docs/MASTER_ROADMAP.md`, `docs/DECISION_LOG.md` and
`docs/EXECUTION_HANDOFF.md` at meaningful milestones. Each completed stage needs a
meaningful local commit, an evidence run and an honest Go/No-Go conclusion.
